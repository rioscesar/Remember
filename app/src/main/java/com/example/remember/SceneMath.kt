package com.example.remember

import kotlin.math.cos
import kotlin.math.sin

data class ProjectedPoint(val x: Float, val y: Float, val depth: Float)

object SceneMath {
    /**
     * COLMAP world axes are X right, Y down, Z forward, while the renderer draws Y upward.
     * Rotating 180 degrees about X converts between them; negating Y alone would mirror the scene.
     */
    fun toViewSpace(x: Float, y: Float, z: Float): Triple<Float, Float, Float> = Triple(x, -y, -z)

    fun project(point: EvidencePoint, yaw: Float, pitch: Float): ProjectedPoint {
        val cosYaw = cos(yaw)
        val sinYaw = sin(yaw)
        val x = point.x * cosYaw - point.z * sinYaw
        val z = point.x * sinYaw + point.z * cosYaw
        val cosPitch = cos(pitch)
        val sinPitch = sin(pitch)
        return ProjectedPoint(x, point.y * cosPitch - z * sinPitch, point.y * sinPitch + z * cosPitch)
    }
}
