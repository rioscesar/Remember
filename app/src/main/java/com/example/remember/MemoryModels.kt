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
)

data class EvidencePoint(
    val x: Float,
    val y: Float,
    val z: Float,
    val red: Int,
    val green: Int,
    val blue: Int,
    val support: Int,
    val reprojectionError: Float,
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
                        put("points", JSONArray().apply {
                            reconstruction.points.forEach { point ->
                                put(JSONObject().apply {
                                    put("x", point.x); put("y", point.y); put("z", point.z)
                                    put("r", point.red); put("g", point.green); put("b", point.blue)
                                    put("support", point.support)
                                    put("reprojectionError", point.reprojectionError)
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
                        reprojectionError = it.getDouble("reprojectionError").toFloat(),
                    )
                }
            },
            registeredPhotoNames = getJSONArray("registeredPhotoNames").toStrings(),
            rejectedPhotoNames = getJSONArray("rejectedPhotoNames").toStrings(),
        )
    }

    private fun JSONArray.toStrings(): List<String> = List(length()) { getString(it) }
}
