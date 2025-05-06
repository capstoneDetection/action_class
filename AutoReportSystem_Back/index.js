import express from "express";
import cors from "cors";

import authRoutes from "./routes/api/auth.js";
import userStatusRoutes from "./routes/api/userStatus.js";
import userRoutes from "./routes/api/user.js";
import smsRouter from "./routes/api/sms.js";

const app = express();

app.use(express.json());
app.use(cors());

app.use("/api/auth", authRoutes);
app.use("/api/userstatus", userStatusRoutes);
app.use("/api/user", userRoutes);
app.use("/api/sms", smsRouter);

app.listen(8080, () => {
  console.log("8080port Server Open");
});
