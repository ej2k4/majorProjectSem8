import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:5000",
});

export const generateStory = (data) =>
  API.post("/generate", data);

export const predictASD = (data) =>
  API.post("/predict", data);

export const generateCartoon = (formData) =>
  API.post("/cartoon", formData);