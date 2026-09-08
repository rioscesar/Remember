package com.example.remember

import org.json.JSONArray
import org.json.JSONObject

object ReconstructionBundle {
    private const val VERSION = 1

    fun parse(json: String): Reconstruction {
        val root = JSONObject(json)
        require(root.getInt("formatVersion") == VERSION) { "This reconstruction was made by an unsupported companion version." }
        val sourceImages = root.getJSONArray("sourceImages").toStrings()
        require(sourceImages.distinct().size >= 3) { "The reconstruction bundle does not contain enough source evidence." }
        val registeredPhotoNames = root.getJSONArray("registeredPhotoNames").toStrings()
        require(registeredPhotoNames.distinct().size >= 3) { "Too few photographs contributed to this reconstruction." }
        require(registeredPhotoNames.all(sourceImages::contains)) {
            "The reconstruction bundle has invalid source evidence."
        }
        val points = root.getJSONArray("points").let { array ->
            List(array.length()) { index ->
                array.getJSONObject(index).let {
                    EvidencePoint(
                        x = it.getDouble("x").toFloat(), y = it.getDouble("y").toFloat(),
                        z = it.getDouble("z").toFloat(), red = it.getInt("r"),
                        green = it.getInt("g"), blue = it.getInt("b"),
                        support = it.getInt("support"),
                        reprojectionError = it.getDouble("reprojectionError").toFloat(),
                    )
                }
            }
        }
        require(points.size >= 100) { "This place did not produce enough shared spatial evidence." }
        require(points.all { it.support >= 3 && it.reprojectionError >= 0f }) {
            "The reconstruction contains points without sufficient photographic support."
        }
        return Reconstruction(
            points = points,
            registeredPhotoNames = registeredPhotoNames,
            rejectedPhotoNames = root.optJSONArray("rejectedPhotoNames")?.toStrings().orEmpty(),
        )
    }

    private fun JSONArray.toStrings(): List<String> = List(length()) { getString(it) }
}
