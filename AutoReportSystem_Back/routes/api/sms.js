import express from "express";
import solapi from "solapi";
import axios from "axios";
import pool from "../../DB/db.js";

const router = express.Router();

router.post("/:id", async (req, res) => {
  const { id } = req.params;
  if (!id) return res.status(400).json({ error: "id 값을 입력하세요." });

  try {
    const [rows] = await pool.query(
      "SELECT id, name, status, phoneNumber, emergencyNumber, birth, nkda, disease, address FROM users WHERE id = ?;",
      [id]
    );

    if (rows.length === 0) {
      return res.status(404).json({ error: "유저 정보를 찾을 수 없습니다." });
    }

    const user = rows[0];

    const { SolapiMessageService } = solapi;

    const messageService = new SolapiMessageService(
      "NCSBBSAEWBKI8HDA",
      "ZY9GEDQZAFAQNA7TMD9A9SYEEC78TALI"
    );
    if (user.status === 0) {
      const { address } = req.body;
      await messageService.send({
        to: user.emergencyNumber.replace(/[^0-9]/g, ""),
        from: "01030061194",
        text: `${user.name}님이 위급상황입니다. 생년월일:${user.birth}, 약물 알러지:${user.nkda}, 기저질환:${user.disease}, 위치:${address} `,
      });
    } else {
      await messageService.send({
        to: user.emergencyNumber.replace(/[^0-9]/g, ""),
        from: "01030061194",
        text: `${user.name}님이 위급상황입니다. 생년월일:${user.birth}, 약물 알러지:${user.nkda}, 기저질환:${user.disease}, 위치:${user.address} `,
      });
    }

    res.status(200).json({ message: "문자 전송 성공" });
  } catch (err) {
    res.status(500).json({ message: "문자 전송 실패" });
  }
});

export default router;
