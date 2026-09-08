package com.example.remember

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.json.JSONException
import java.io.IOException

class MainActivity : AppCompatActivity() {
    private lateinit var store: MemoryStore
    private lateinit var content: LinearLayout
    private var pendingMemoryId: String? = null

    private val selectPhotos = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        val memoryId = pendingMemoryId ?: return@registerForActivityResult
        pendingMemoryId = null
        if (uris.size < 3) {
            showMessage("Choose at least three photos of the same place.")
            return@registerForActivityResult
        }
        if (!uris.all(::persistReadPermission)) return@registerForActivityResult
        val memory = store.all().firstOrNull { it.id == memoryId } ?: return@registerForActivityResult
        val oldUris = memory.photos.map { Uri.parse(it.uri) }
        store.save(memory.copy(photos = uris.map { SourcePhoto(it.toString(), displayName(it)) }, reconstruction = null))
        releaseUnusedPermissions(oldUris)
        showMemory(memoryId)
    }

    private val importReconstruction = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val memoryId = pendingMemoryId ?: return@registerForActivityResult
        pendingMemoryId = null
        if (uri == null) return@registerForActivityResult
        val json = try {
            contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                ?: throw IOException("The reconstruction file could not be read.")
        } catch (exception: IOException) {
            showMessage(exception.message ?: "This reconstruction file could not be read.")
            return@registerForActivityResult
        } catch (exception: SecurityException) {
            showMessage("Remember no longer has permission to read this reconstruction file.")
            return@registerForActivityResult
        }
        try {
            val memory = store.all().first { it.id == memoryId }
            val selectedNames = memory.photos.map(SourcePhoto::displayName).toSet()
            val reconstruction = ReconstructionBundle.parse(json)
            require(selectedNames.intersect(reconstruction.registeredPhotoNames.toSet()).size >= 3) {
                "At least three selected photos must contribute to this reconstruction."
            }
            store.save(memory.copy(reconstruction = reconstruction))
            showMemory(memoryId)
        } catch (exception: JSONException) {
            showMessage("This is not a valid Remember reconstruction file.")
        } catch (exception: IllegalArgumentException) {
            showMessage(exception.message ?: "This reconstruction could not be imported.")
        } catch (exception: NoSuchElementException) {
            showMessage("This memory no longer exists.")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = MemoryStore(this)
        pendingMemoryId = savedInstanceState?.getString(PENDING_MEMORY_ID)
        showLibrary()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(PENDING_MEMORY_ID, pendingMemoryId)
    }

    private fun showLibrary() {
        content = screen()
        content.addView(title("Remember"))
        content.addView(body("Places held in your own photographs. Nothing leaves this device."))
        content.addView(button("Create a memory") { askForName() })
        val memories = store.all()
        if (memories.isEmpty()) {
            content.addView(body("Begin with a small set of overlapping photos of one place."))
        } else {
            content.addView(section("Your memories"))
            memories.forEach { memory ->
                content.addView(button(memory.name) { showMemory(memory.id) })
                content.addView(body(memoryStatus(memory)))
            }
        }
        showContent()
    }

    private fun askForName() {
        val input = EditText(this).apply { hint = "A name for this place" }
        MaterialAlertDialogBuilder(this)
            .setTitle("Create a memory")
            .setView(input)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Choose photos") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isBlank()) {
                    showMessage("Give this memory a name first.")
                } else {
                    val memory = Memory(name = name, photos = emptyList())
                    store.save(memory)
                    pendingMemoryId = memory.id
                    selectPhotos.launch(arrayOf("image/*"))
                }
            }
            .show()
    }

    private fun showMemory(id: String) {
        val memory = store.all().firstOrNull { it.id == id } ?: run { showLibrary(); return }
        content = screen()
        content.addView(title(memory.name))
        content.addView(body("${memory.photos.size} selected photographs. Original files remain where you keep them."))
        content.addView(button("Choose different photos") {
            pendingMemoryId = memory.id
            selectPhotos.launch(arrayOf("image/*"))
        })
        if (memory.reconstruction == null) {
            content.addView(section("Putting this place back together…"))
            content.addView(body("This first version uses the private Remember Companion on your computer to recover only geometry supported by your selected photos. Import its reconstruction here when it is ready."))
            content.addView(button("Import local reconstruction") {
                pendingMemoryId = memory.id
                importReconstruction.launch(arrayOf("application/json", "text/json"))
            })
            content.addView(body("If the photos do not overlap enough, the companion will say so. Remember will not fill in what the photographs cannot support."))
        } else {
            content.addView(section("A preserved place"))
            content.addView(body("${memory.reconstruction.points.size} spatial points supported by ${memory.reconstruction.registeredPhotoNames.size} photographs. Where support thins out, the place fades."))
            content.addView(button("Enter") { showScene(memory) })
        }
        content.addView(button("Delete this memory") { confirmDelete(memory) })
        content.addView(button("Back to library") { showLibrary() })
        showContent()
    }

    private fun showScene(memory: Memory) {
        val reconstruction = memory.reconstruction ?: return
        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        layout.addView(EvidenceSceneView(this, reconstruction.points), LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f,
        ))
        layout.addView(body("Drag to look around. These points are recovered from matching photographs; their gradual disappearance marks where photographic support becomes sparse."))
        layout.addView(button("Leave this place") { showMemory(memory.id) })
        setContentView(layout)
    }

    private fun confirmDelete(memory: Memory) {
        MaterialAlertDialogBuilder(this)
            .setTitle("Delete ${memory.name}?")
            .setMessage("This removes Remember's local photo references and reconstruction. It never deletes your original photos.")
            .setNegativeButton("Keep", null)
            .setPositiveButton("Delete") { _, _ ->
                val oldUris = memory.photos.map { Uri.parse(it.uri) }
                store.delete(memory.id)
                releaseUnusedPermissions(oldUris)
                showLibrary()
            }
            .show()
    }

    private fun screen(): LinearLayout {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 72, 48, 48)
        }
        return column.also { column ->
            val scroll = ScrollView(this)
            scroll.addView(column)
            column.tag = scroll
        }
    }

    private fun title(value: String) = TextView(this).apply {
        text = value; textSize = 32f; setPadding(0, 0, 0, 20)
    }

    private fun section(value: String) = TextView(this).apply {
        text = value; textSize = 20f; setPadding(0, 36, 0, 12)
    }

    private fun body(value: String) = TextView(this).apply {
        text = value; textSize = 16f; setPadding(0, 0, 0, 20)
    }

    private fun button(value: String, action: () -> Unit) = Button(this).apply {
        text = value; isAllCaps = false; gravity = Gravity.START; setOnClickListener { action() }
    }

    private fun showContent() {
        super.setContentView((content.tag as ScrollView))
    }

    private fun persistReadPermission(uri: Uri): Boolean {
        try {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            return true
        } catch (exception: SecurityException) {
            showMessage("Remember could not retain access to ${displayName(uri)}.")
            return false
        }
    }

    private fun releaseUnusedPermissions(uris: List<Uri>) {
        val inUse = store.all().flatMap { it.photos }.map { it.uri }.toSet()
        uris.filterNot { it.toString() in inUse }.forEach { uri ->
            try {
                contentResolver.releasePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (exception: SecurityException) {
                showMessage("Remember could not release access to ${displayName(uri)}. Remove it in system settings.")
            }
        }
    }

    private fun displayName(uri: Uri): String = contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        cursor.moveToFirst()
        cursor.getString(cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME))
    } ?: uri.lastPathSegment ?: "Selected photo"

    private fun memoryStatus(memory: Memory): String =
        if (memory.reconstruction == null) "Awaiting local reconstruction" else "Ready to enter"

    private fun showMessage(message: String) {
        AlertDialog.Builder(this).setMessage(message).setPositiveButton("OK", null).show()
    }

    private companion object {
        const val PENDING_MEMORY_ID = "pending_memory_id"
    }
}
