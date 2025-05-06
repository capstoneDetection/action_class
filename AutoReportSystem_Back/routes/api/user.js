import express from "express";
import pool from "../../DB/db.js";
import authMiddleware from "../../middleware/auth.js";

const router = express.Router();

router.get("/:id", authMiddleware, async (req, res) => {
  const { id } = req.params;
  if (!id) {
    return res.status(400).json({ error: "id 값을 입력하세요." });
  }

  try {
    const [rows] = await pool.query(
      "SELECT id, name, phoneNumber, emergencyNumber, birth, nkda, disease, address FROM users WHERE id = ?;",
      [id]
    );

    if (rows.length === 0) {
      return res
        .status(404)
        .json({ error: "해당 ID의 데이터가 존재하지 않습니다." });
    }

    res.json({ data: rows[0] });
  } catch (error) {
    console.error("DB 조회 실패:", error);
    res.status(500).json({ error: "DB 조회 실패: " + error.message });
  }
});

export default router;
