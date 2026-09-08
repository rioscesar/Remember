package com.example.remember

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Renders recovered evidence points. When the companion supplied recovered camera poses the
 * viewer stands where a photograph was actually taken and looks around from there, which is a
 * position the reconstruction solved for rather than an invented vantage point.
 */
class EvidenceSceneView(
    context: Context,
    private val points: List<EvidencePoint>,
    private val cameraPoses: List<CameraPose> = emptyList(),
    private val focalLengthNormalized: Float? = null,
) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var yaw = 0f
    private var pitch = 0f
    private var lastX = 0f
    private var lastY = 0f
    private var viewpoint = 0

    private val centerX = (points.minOf { it.x } + points.maxOf { it.x }) / 2f
    private val centerY = (points.minOf { it.y } + points.maxOf { it.y }) / 2f
    private val centerZ = (points.minOf { it.z } + points.maxOf { it.z }) / 2f
    private val extent = points.maxOf {
        max(abs(it.x - centerX), max(abs(it.y - centerY), abs(it.z - centerZ)))
    }.coerceAtLeast(0.001f)

    /** True when the scene can be viewed from a position a photograph was taken from. */
    private val standsInPlace: Boolean get() = cameraPoses.isNotEmpty()

    // Point attributes are flattened once so rasterising a frame touches primitive arrays
    // instead of walking a list of objects several hundred thousand times.
    private val count = points.size
    private val px = FloatArray(count) { points[it].x }
    private val py = FloatArray(count) { points[it].y }
    private val pz = FloatArray(count) { points[it].z }
    private val argb = IntArray(count) {
        val point = points[it]
        Color.argb(evidenceAlpha(point), point.red, point.green, point.blue)
    }

    private var frame: Bitmap? = null
    private var pixels = IntArray(0)
    private var depthBuffer = FloatArray(0)

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (w <= 0 || h <= 0) return
        frame = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        pixels = IntArray(w * h)
        depthBuffer = FloatArray(w * h)
    }

    val viewpointLabel: String
        get() = if (standsInPlace) "Standing at photograph ${viewpoint + 1} of ${cameraPoses.size}" else ""

    fun nextViewpoint() {
        if (!standsInPlace) return
        viewpoint = (viewpoint + 1) % cameraPoses.size
        yaw = 0f
        pitch = 0f
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(12, 12, 14))
        if (standsInPlace) drawFromRecoveredViewpoint(canvas) else drawOrbiting(canvas)
    }

    private fun drawFromRecoveredViewpoint(canvas: Canvas) {
        val target = frame ?: return
        val pose = cameraPoses[viewpoint.coerceIn(cameraPoses.indices)]
        val rotation = SceneMath.rotationRows(pose)
        val focal = (focalLengthNormalized ?: DEFAULT_FOCAL) * width
        val cosYaw = cos(yaw)
        val sinYaw = sin(yaw)
        val cosPitch = cos(pitch)
        val sinPitch = sin(pitch)
        val halfWidth = width / 2f
        val halfHeight = height / 2f

        java.util.Arrays.fill(pixels, BACKGROUND)
        java.util.Arrays.fill(depthBuffer, Float.MAX_VALUE)

        for (index in 0 until count) {
            val (cx, cy, cz) = SceneMath.toCameraSpace(px[index], py[index], pz[index], rotation, pose)
            // Look around from the photographer's position. Camera space is X right,
            // Y down, Z forward, so yaw turns about Y and pitch about X.
            val rx = cx * cosYaw + cz * sinYaw
            val rz = -cx * sinYaw + cz * cosYaw
            val ry = cy * cosPitch - rz * sinPitch
            val depth = cy * sinPitch + rz * cosPitch
            if (depth <= NEAR_PLANE) continue

            val screenX = (halfWidth + focal * rx / depth).roundToInt()
            val screenY = (halfHeight + focal * ry / depth).roundToInt()
            // A dense sample stands for a small patch of surface, so its footprint shrinks
            // with distance rather than being a fixed dot.
            val radius = (focal * POINT_WORLD_RADIUS / depth).roundToInt().coerceIn(1, MAX_SPLAT_RADIUS)
            if (screenX + radius < 0 || screenX - radius >= width) continue
            if (screenY + radius < 0 || screenY - radius >= height) continue

            val colour = argb[index]
            for (offsetY in -radius..radius) {
                val y = screenY + offsetY
                if (y < 0 || y >= height) continue
                val row = y * width
                for (offsetX in -radius..radius) {
                    val x = screenX + offsetX
                    if (x < 0 || x >= width) continue
                    val slot = row + x
                    // Nearer evidence hides what is behind it, so walls occlude instead of
                    // every point in the place showing through at once.
                    if (depth >= depthBuffer[slot]) continue
                    depthBuffer[slot] = depth
                    pixels[slot] = blendOverBackground(colour)
                }
            }
        }

        target.setPixels(pixels, 0, width, 0, 0, width, height)
        canvas.drawBitmap(target, 0f, 0f, null)
    }

    /** Composites an evidence colour onto the empty-space background at its own confidence. */
    private fun blendOverBackground(colour: Int): Int {
        val alpha = Color.alpha(colour)
        if (alpha >= 255) return colour or OPAQUE
        val inverse = 255 - alpha
        val red = (Color.red(colour) * alpha + BACKGROUND_RED * inverse) / 255
        val green = (Color.green(colour) * alpha + BACKGROUND_GREEN * inverse) / 255
        val blue = (Color.blue(colour) * alpha + BACKGROUND_BLUE * inverse) / 255
        return Color.rgb(red, green, blue) or OPAQUE
    }

    private fun drawOrbiting(canvas: Canvas) {
        val scale = min(width, height) * 0.42f
        points.map { point ->
            val (x, y, z) = SceneMath.toViewSpace(
                (point.x - centerX) / extent,
                (point.y - centerY) / extent,
                (point.z - centerZ) / extent,
            )
            point to SceneMath.project(point.copy(x = x, y = y, z = z), yaw, pitch)
        }
            .sortedBy { (_, projected) -> projected.depth }
            .forEach { (point, projected) ->
                paint.color = Color.rgb(point.red, point.green, point.blue)
                paint.alpha = evidenceAlpha(point)
                val perspective = 4.5f / (4.5f + projected.depth)
                canvas.drawCircle(
                    width / 2f + projected.x * scale * perspective,
                    height / 2f - projected.y * scale * perspective,
                    (1.2f + point.support * 0.16f) * perspective,
                    paint,
                )
            }
    }

    /**
     * Every retained point already cleared the evidence threshold, so it is shown. Additional
     * supporting photographs still read as more certain, but the weakest accepted evidence stays
     * legible instead of fading to near-invisibility.
     */
    private fun evidenceAlpha(point: EvidencePoint): Int {
        val support = (150 + (point.support - 3) * 26).coerceIn(150, 255)
        val quality = point.reprojectionError?.let { (255 - it * 30).toInt().coerceIn(120, 255) } ?: 255
        return min(support, quality)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> { lastX = event.x; lastY = event.y; return true }
            MotionEvent.ACTION_MOVE -> {
                yaw += (event.x - lastX) / width * LOOK_SPEED
                pitch = (pitch + (event.y - lastY) / height * LOOK_SPEED).coerceIn(-1.2f, 1.2f)
                lastX = event.x; lastY = event.y; invalidate()
            }
        }
        return true
    }

    private companion object {
        const val NEAR_PLANE = 0.05f
        const val DEFAULT_FOCAL = 0.9f
        const val POINT_WORLD_RADIUS = 0.012f
        const val LOOK_SPEED = 2.2f
        const val MAX_SPLAT_RADIUS = 6
        const val OPAQUE = 0xFF000000.toInt()
        const val BACKGROUND_RED = 12
        const val BACKGROUND_GREEN = 12
        const val BACKGROUND_BLUE = 14
        val BACKGROUND = Color.rgb(BACKGROUND_RED, BACKGROUND_GREEN, BACKGROUND_BLUE) or OPAQUE
    }
}
