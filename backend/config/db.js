import mongoose from 'mongoose';
import dotenv from 'dotenv';
dotenv.config();

export const connectToMongoDB = async () => {
    try {
        const MONGO_URI = process.env.MONGO_URI;
        await mongoose.connect(MONGO_URI)
        console.log('Connected to MongoDB successfully!');
    } catch (error) {
        console.error('Error connecting to MongoDB:', error);
        throw error;
    }
};