package com.example.remember

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

data class Memory(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val createdAt: Long = System.currentTimeMillis(),
    val photos: List<SourcePhoto>,
    val reconstruction: Reconstruction? = null,
)

data class SourcePhoto(val uri: String, val displayName: String)

data class Reconstruction(
    val points: List<EvidencePoint>,
    val registeredPhotoNames: List<String>,
    val rejectedPhotoNames: List<String>,
    val cameraPoses: List<CameraPose> = emptyList(),
    val focalLengthNormalized: Float? = null,
)

/**
 * A position a photograph was actually taken from, recovered by the companion.
 * Rotation is world-to-camera, matching COLMAP's convention.
 */
data class CameraPose(
    val name: String,
    val qw: Float,
    val qx: Float,
    val qy: Float,
    val qz: Float,
    val tx: Float,
    val ty: Float,
    val tz: Float,
)

data class EvidencePoint(
    val x: Float,
    val y: Float,
    val z: Float,
    val red: Int,
    val green: Int,
    val blue: Int,
    val support: Int,
    val reprojectionError: Float? = null,
)

object MemoryJson {
    fun encode(memories: List<Memory>): String = JSONArray().apply {
        memories.forEach { memory ->
            put(JSONObject().apply {
                put("id", memory.id)
                put("name", memory.name)
                put("createdAt", memory.createdAt)
                put("photos", JSONArray().apply {
                    memory.photos.forEach { photo ->
                        put(JSONObject().put("uri", photo.uri).put("displayName", photo.displayName))
                    }
                })
                memory.reconstruction?.let { reconstruction ->
                    put("reconstruction", JSONObject().apply {
                        put("registeredPhotoNames", JSONArray(reconstruction.registeredPhotoNames))
                        put("rejectedPhotoNames", JSONArray(reconstruction.rejectedPhotoNames))
                        put("focalLengthNormalized", reconstruction.focalLengthNormalized ?: JSONObject.NULL)
                        put("cameraPoses", JSONArray().apply {
                            reconstruction.cameraPoses.forEach { pose ->
                                put(JSONObject().apply {
                                    put("name", pose.name)
                                    put("qw", pose.qw); put("qx", pose.qx)
                                    put("qy", pose.qy); put("qz", pose.qz)
                                    put("tx", pose.tx); put("ty", pose.ty); put("tz", pose.tz)
                                })
                            }
                        })
                        put("points", JSONArray().apply {
                            reconstruction.points.forEach { point ->
                                put(JSONObject().apply {
                                    put("x", point.x); put("y", point.y); put("z", point.z)
                                    put("r", point.red); put("g", point.green); put("b", point.blue)
                                    put("support", point.support)
                                    put("reprojectionError", point.reprojectionError ?: JSONObject.NULL)
                                })
                            }
                        })
                    })
                }
            })
        }
    }.toString()

    fun decode(value: String): List<Memory> = JSONArray(value).let { array ->
        List(array.length()) { index ->
            val item = array.getJSONObject(index)
            val photos = item.getJSONArray("photos").let { photoArray ->
                List(photoArray.length()) { photoIndex ->
                    photoArray.getJSONObject(photoIndex).let {
                        SourcePhoto(it.getString("uri"), it.getString("displayName"))
                    }
                }
            }
            Memory(
                id = item.getString("id"),
                name = item.getString("name"),
                createdAt = item.getLong("createdAt"),
                photos = photos,
                reconstruction = item.optJSONObject("reconstruction")?.toReconstruction(),
            )
        }
    }

    private fun JSONObject.toReconstruction(): Reconstruction {
        val pointArray = getJSONArray("points")
        return Reconstruction(
            points = List(pointArray.length()) { index ->
                pointArray.getJSONObject(index).let {
                    EvidencePoint(
                        x = it.getDouble("x").toFloat(), y = it.getDouble("y").toFloat(),
                        z = it.getDouble("z").toFloat(), red = it.getInt("r"),
                        green = it.getInt("g"), blue = it.getInt("b"), support = it.getInt("support"),
                        reprojectionError = if (it.isNull("reprojectionError")) null else it.getDouble("reprojectionError").toFloat(),
                    )
                }
            },
            registeredPhotoNames = getJSONArray("registeredPhotoNames").toStrings(),
            rejectedPhotoNames = getJSONArray("rejectedPhotoNames").toStrings(),
            cameraPoses = optJSONArray("cameraPoses")?.let { array ->
                List(array.length()) { index -> array.getJSONObject(index).toCameraPose() }
            }.orEmpty(),
            focalLengthNormalized = if (isNull("focalLengthNormalized")) null else getDouble("focalLengthNormalized").toFloat(),
        )
    }

    private fun JSONObject.toCameraPose(): CameraPose = CameraPose(
        name = getString("name"),
        qw = getDouble("qw").toFloat(), qx = getDouble("qx").toFloat(),
        qy = getDouble("qy").toFloat(), qz = getDouble("qz").toFloat(),
        tx = getDouble("tx").toFloat(), ty = getDouble("ty").toFloat(), tz = getDouble("tz").toFloat(),
    )

    private fun JSONArray.toStrings(): List<String> = List(length()) { getString(it) }
}
