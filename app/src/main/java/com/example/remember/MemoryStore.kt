package com.example.remember

import android.content.Context

class MemoryStore(context: Context) {
    private val preferences = context.getSharedPreferences("remember_memories", Context.MODE_PRIVATE)

    fun all(): List<Memory> = preferences.getString(MEMORIES_KEY, "[]")
        ?.let(MemoryJson::decode)
        .orEmpty()
        .sortedByDescending(Memory::createdAt)

    fun save(memory: Memory) {
        val updated = all().filterNot { it.id == memory.id } + memory
        preferences.edit().putString(MEMORIES_KEY, MemoryJson.encode(updated)).apply()
    }

    fun delete(id: String) {
        preferences.edit().putString(MEMORIES_KEY, MemoryJson.encode(all().filterNot { it.id == id })).apply()
    }

    private companion object {
        const val MEMORIES_KEY = "memories"
    }
}
