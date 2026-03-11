import express from 'express';
import { createJournalEntry, deleteJournalEntry, getJournal, updateJournalEntry } from '../controllers/journal.controller.js';

const router = express.Router();



router.get('/', getJournal);  

router.post('/', createJournalEntry);

router.put('/:id', updateJournalEntry);

router.delete('/:id', deleteJournalEntry);




export default router;