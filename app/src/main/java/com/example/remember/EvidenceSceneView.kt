package com.example.remember

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
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
        val pose = cameraPoses[viewpoint.coerceIn(cameraPoses.indices)]
        val rotation = SceneMath.rotationRows(pose)
        val focal = (focalLengthNormalized ?: DEFAULT_FOCAL) * width
        val cosYaw = cos(yaw)
        val sinYaw = sin(yaw)
        val cosPitch = cos(pitch)
        val sinPitch = sin(pitch)
        data class Sample(val point: EvidencePoint, val x: Float, val y: Float, val depth: Float)

        points.asSequence()
            .map { point ->
                val (cx, cy, cz) = SceneMath.toCameraSpace(point.x, point.y, point.z, rotation, pose)
                val rx = cx * cosYaw + cz * sinYaw
                val rz = -cx * sinYaw + cz * cosYaw
                val ry = cy * cosPitch - rz * sinPitch
                val depth = cy * sinPitch + rz * cosPitch
                Sample(point, rx, ry, depth)
            }
            .filter { it.depth > NEAR_PLANE }
            .sortedByDescending { it.depth }
            .forEach { sample ->
                paint.color = Color.rgb(sample.point.red, sample.point.green, sample.point.blue)
                paint.alpha = evidenceAlpha(sample.point)
                val radius = (focal * POINT_WORLD_RADIUS / sample.depth).coerceIn(1f, 14f)
                canvas.drawCircle(
                    width / 2f + focal * sample.x / sample.depth,
                    height / 2f + focal * sample.y / sample.depth,
                    radius,
                    paint,
                )
            }
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
    }
}
