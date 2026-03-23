import express from 'express';
import { createJournalEntry, deleteJournalEntry, getJournal, updateJournalEntry } from '../controllers/journal.controller.js';

const router = express.Router();



router.get('/getAll', getJournal);  

router.post('/create', createJournalEntry);

router.put('/edit/:id', updateJournalEntry);

router.delete('/del/:id', deleteJournalEntry);




export default router;