package com.example.remember

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class EvidenceSceneView(context: Context, private val points: List<EvidencePoint>) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var yaw = 0.35f
    private var pitch = -0.18f
    private var lastX = 0f
    private var lastY = 0f
    private val centerX = (points.minOf { it.x } + points.maxOf { it.x }) / 2f
    private val centerY = (points.minOf { it.y } + points.maxOf { it.y }) / 2f
    private val centerZ = (points.minOf { it.z } + points.maxOf { it.z }) / 2f
    private val extent = points.maxOf {
        max(abs(it.x - centerX), max(abs(it.y - centerY), abs(it.z - centerZ)))
    }.coerceAtLeast(0.001f)

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(20, 20, 18))
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
                val supportAlpha = ((point.support - 2) * 48).coerceIn(48, 190)
                val qualityAlpha = point.reprojectionError
                    ?.let { (190 - it * 35).toInt().coerceIn(60, 190) }
                    ?: 190
                paint.color = Color.rgb(point.red, point.green, point.blue)
                paint.alpha = min(supportAlpha, qualityAlpha)
                val perspective = 4.5f / (4.5f + projected.depth)
                canvas.drawCircle(
                    width / 2f + projected.x * scale * perspective,
                    height / 2f - projected.y * scale * perspective,
                    (1.2f + point.support * 0.16f) * perspective,
                    paint,
                )
            }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> { lastX = event.x; lastY = event.y; return true }
            MotionEvent.ACTION_MOVE -> {
                yaw += (event.x - lastX) / width
                pitch = (pitch + (event.y - lastY) / height).coerceIn(-1.2f, 1.2f)
                lastX = event.x; lastY = event.y; invalidate()
            }
        }
        return true
    }
}
