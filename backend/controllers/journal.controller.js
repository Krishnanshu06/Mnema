import mongoose from "mongoose";
import path from "path";
import Journal from "../models/journal.model.js";

const __dirname = import.meta.dirname;

export const getJournal = async(req, res) => {
  
  try {
    const allUsers = await Journal.find({});
    return res.status(200).json({ success: true, data: allUsers });
  }  catch (err) {
    return res.status(500).json({ success: false, data: null });
  }
    
};

export const createJournalEntry = async (req, res) => {
  console.log(req.body);

  const { title, content, userId, attachments } = req.body;

  if (!title || !content || !userId) {
    return res.status(400).json({
      error: "Title, content, and userId are required fields.",
      providedData: { title, content, userId, attachments },
    });
  }

  const journal = req.body;
  const journalEntry = new Journal(journal);

  try {
    console.log(journalEntry);
    await journalEntry.save();
    res.status(201).json({ success: true, data: journalEntry });
  } catch (error) {
    console.error("Error saving journal entry:", error);
  }
};

export const updateJournalEntry = async (req, res) => {
  const MongoId = req.params.id;
  /// to update any field
};

export const deleteJournalEntry = async (req, res) => {
  const MongoId = req.params.id;

  try {
    await Journal.findByIdAndDelete(MongoId);
    res.status(200).json({ success: true ,message: "Journal entry deleted successfully." });
    console.log("Deleted Entry by id:", MongoId);
  } catch (error) {
    console.error("Error deleting journal entry:", error);
  }
};
