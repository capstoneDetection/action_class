import mysql from "mysql2/promise";

const pool = mysql.createPool({
  host: "svc.sel4.cloudtype.app",
  port: "31258",
  user: "root",
  password: "m8u7dmtw4dc6206d",
  database: "appdb",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

export default pool;
