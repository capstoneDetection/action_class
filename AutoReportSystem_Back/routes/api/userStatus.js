import express from "express";
import axios from "axios";
import pool from "../../DB/db.js";

const router = express.Router();

router.post("/status", async (req, res) => {
  const { id, status } = req.body;

  if (!id || status === undefined) {
    return res.status(400).json({ error: "정확한 값이 아닙니다" });
  }

  try {
    const [result] = await pool.query("UPDATE users SET status=? WHERE id=?", [
      status,
      id,
    ]);

    if (result.affectedRows === 0) {
      return res.status(404).json({
        error: "해당 ID의 데이터가 존재하지 않습니다.",
      });
    }
    if (status === 1) {
      await axios.post(`http://localhost:8080/api/sms/${id}`);
      await pool.query("UPDATE users SET status=? WHERE id=?", [0, id]);
    }
    res.status(200).json({ message: "상태 변경 성공" });
  } catch (error) {
    console.error("상태 변경 실패:", error);
    res.status(500).json({ error: "상태 변경 실패: " + error.message });
  }
});

router.get("/:device", async (req, res) => {
  const { device } = req.params;

  if (!device) {
    return res.status(400).json({ error: "등록 기기가 없습니다." });
  }

  try {
    const [rows] = await pool.query("SELECT id FROM users WHERE device = ?", [
      device,
    ]);

    if (rows.length === 0) {
      return res.status(404).json({ error: "연결 기기가 없습니다." });
    }

    res.json({ id: rows[0].id });
  } catch (error) {
    console.error("DB 조회 실패:", error);
    res.status(500).json({ error: "DB 조회 실패: " + error.message });
  }
});

export default router;
