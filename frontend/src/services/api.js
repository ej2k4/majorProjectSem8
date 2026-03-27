import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const generateFull = (data) =>
  API.post("/generate-full", data);

export const predictSentence = (data) =>
  API.post("/predict-sentence", data);