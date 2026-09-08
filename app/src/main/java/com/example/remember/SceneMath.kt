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

    /** World-to-camera rotation rows for a COLMAP quaternion. */
    fun rotationRows(pose: CameraPose): Array<FloatArray> {
        val w = pose.qw
        val x = pose.qx
        val y = pose.qy
        val z = pose.qz
        return arrayOf(
            floatArrayOf(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            floatArrayOf(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            floatArrayOf(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    }

    /**
     * Place a world point in the frame of a photograph that was actually taken,
     * so the viewer looks out from where the photographer stood.
     */
    fun toCameraSpace(
        x: Float,
        y: Float,
        z: Float,
        rotation: Array<FloatArray>,
        pose: CameraPose,
    ): Triple<Float, Float, Float> = Triple(
        rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z + pose.tx,
        rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z + pose.ty,
        rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z + pose.tz,
    )
}
