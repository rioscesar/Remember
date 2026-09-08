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

    @Test
    fun `view space rights COLMAP y-down axes without mirroring`() {
        val (x, y, z) = SceneMath.toViewSpace(1f, 2f, 3f)

        assertEquals(1f, x, 0.001f)
        assertEquals(-2f, y, 0.001f)
        assertEquals(-3f, z, 0.001f)
    }

    @Test
    fun `view space conversion is its own inverse`() {
        val (x, y, z) = SceneMath.toViewSpace(1f, 2f, 3f).let { SceneMath.toViewSpace(it.first, it.second, it.third) }

        assertEquals(1f, x, 0.001f)
        assertEquals(2f, y, 0.001f)
        assertEquals(3f, z, 0.001f)
    }
}
