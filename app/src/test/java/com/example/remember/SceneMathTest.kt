package com.example.remember

import org.junit.Assert.assertEquals
import org.junit.Test

class SceneMathTest {
    @Test
    fun `zero rotation preserves coordinates`() {
        val point = EvidencePoint(1f, 2f, 3f, 1, 2, 3, 3, 0.5f)

        val result = SceneMath.project(point, 0f, 0f)

        assertEquals(1f, result.x, 0.001f)
        assertEquals(2f, result.y, 0.001f)
        assertEquals(3f, result.depth, 0.001f)
    }
}
