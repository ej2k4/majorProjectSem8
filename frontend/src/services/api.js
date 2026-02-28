import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:5005",
});

export const generateFull = (data) =>
  API.post("/generate-full", data);