const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const fs = require("fs");
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));


dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static("public"));

const inventario = JSON.parse(fs.readFileSync("inventario.json", "utf-8"));

app.post("/api/chat", async (req, res) => {
  const { prompt } = req.body;

  const context = `
Eres un asistente de una clínica.
Tu tarea es responder de manera natural y clara preguntas sobre este inventario:

${JSON.stringify(inventario, null, 2)}

Si el usuario escribe con errores o de forma informal, igual debes entenderlo y responder correctamente.
`;

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "deepseek/deepseek-r1:free"
,
        messages: [
          { role: "system", content: context },
          { role: "user", content: prompt }
        ]
      })
    });

    const data = await response.json();
const output = data.choices?.[0]?.message?.content ||
               data.choices?.[0]?.text ||
               "No se pudo generar una respuesta.";

console.log("✅ RESPUESTA:", output);

res.json({ respuesta: output });

  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error al consultar la IA." });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Servidor escuchando en http://localhost:${PORT}`);
});
