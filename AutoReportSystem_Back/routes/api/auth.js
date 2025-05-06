import express from "express";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";
import pool from "../../DB/db.js";

const router = express.Router();

// 회원가입
router.post("/signup", async (req, res) => {
  const {
    name,
    phoneNumber,
    password,
    emergencyNumber,
    birth,
    nkda,
    disease,
    address,
    device,
  } = req.body;

  if (
    !name ||
    !phoneNumber ||
    !password ||
    !emergencyNumber ||
    !birth ||
    !nkda ||
    !disease ||
    !address ||
    !device
  ) {
    return res.status(400).json({ error: "모든 필드를 입력해주세요." });
  }

  try {
    const hashedPassword = await bcrypt.hash(password, 10);

    await pool.query(
      "INSERT INTO users (name, phoneNumber, password, status, emergencyNumber, birth, nkda, disease, address, device) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      [
        name,
        phoneNumber,
        hashedPassword,
        0,
        emergencyNumber,
        birth,
        nkda,
        disease,
        address,
        device,
      ]
    );

    res.status(201).json({ message: "회원가입 성공!" });
  } catch (error) {
    console.error("회원가입 오류:", error);
    res.status(500).json({ error: "회원가입 실패: " + error.message });
  }
});

// 로그인
router.post("/login", async (req, res) => {
  const { phoneNumber, password } = req.body;

  if (!phoneNumber || !password) {
    return res.status(400).json({ error: "전화번호와 비밀번호를 입력하세요." });
  }

  try {
    const [results] = await pool.query(
      "SELECT * FROM users WHERE phoneNumber = ?",
      [phoneNumber]
    );

    if (results.length === 0) {
      return res.status(401).json({ error: "사용자가 존재하지 않습니다." });
    }

    const user = results[0];
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ error: "비밀번호가 틀렸습니다." });
    }

    const token = jwt.sign({ userId: user.id }, "KimSecret");

    res.status(200).json({ message: "로그인 성공!", token, userId: user.id });
  } catch (error) {
    console.error("로그인 오류:", error);
    res.status(500).json({ error: "서버 오류: " + error.message });
  }
});

export default router;
