import express from 'express'; 
import { connectToMongoDB } from './config/db.js';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const journalRouter = await import('./routes/journal.route.js');

app.use(express.json());



app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.use('/journal', (journalRouter.default));


app.listen(PORT, () => {
    connectToMongoDB()
    console.log('Example app listening on port http://localhost:' + PORT);
});



//keku.database.2006
//mongodb+srv://krishnanshuagrawal:keku.database.2006@data.ttspewy.mongodb.net/?appName=data