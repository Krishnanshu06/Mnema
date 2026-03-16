# Mnema

Mnema is a personal journaling and knowledge management system that helps users store, organize, and query their thoughts, notes, and documents.

The project integrates an AI-powered retrieval system that allows users to ask questions about their stored information and receive context-aware responses generated from their own data.

The long-term goal of Mnema is to build an intelligent personal memory system that can summarize experiences, retrieve past information, and assist users in reflecting on their knowledge and daily life.

---

## Motivation

People capture information across many platforms — notes, journals, voice memos, documents, and emails. Over time this information becomes difficult to navigate and retrieve.

Mnema aims to solve this problem by combining **personal knowledge management** with **AI-assisted retrieval**, allowing users to interact with their stored data in a more natural and intelligent way.

---

## Core Features

- Personal journaling system for storing thoughts and notes
- Document ingestion and storage
- AI-powered querying of stored information
- Retrieval-Augmented Generation (RAG) pipeline
- Context-aware response generation using a local language model
- Structured storage of user data using MongoDB

---

## AI Retrieval System

Mnema implements a **custom Retrieval-Augmented Generation (RAG) pipeline built from scratch** to understand how modern AI knowledge systems work internally.

The pipeline includes:

- Document ingestion
- Text chunking
- Embedding generation
- Vector similarity search
- Context retrieval
- Response generation using a local LLM

Rather than relying on frameworks like LangChain, the system focuses on implementing the core components directly to gain a deeper understanding of AI retrieval architectures.

---

## Tech Stack

**Frontend**
- Vite
- React

**Backend**
- Node.js
- Express

**AI Pipeline**
- Python

**Database**
- MongoDB

---

## Concepts Explored

- Retrieval-Augmented Generation (RAG)
- Vector embeddings
- Similarity search
- Information retrieval
- Personal knowledge management systems
- AI-assisted summarization

---

## Future Features

Mnema is being developed as a long-term project. Planned features include:

- Voice note ingestion
- Automatic summarization of journal entries
- Timeline visualization of personal memories
- Semantic search across stored knowledge
- Integration with external note platforms
- AI-generated insights from journal data

---

## Project Status

Work in progress — additional features and improvements will be added over time.

This project is primarily focused on exploring how AI can augment personal knowledge systems and journaling workflows.

