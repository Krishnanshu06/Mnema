    import { create } from 'zustand'


    const useJournalStore = create((set) => ({

        journals: [],

        createJournal: async (newJournal) => {
            if (!newJournal.title || !newJournal.content) {
                return { success: false, message: "Title and content are required." };
            }

            const res = await fetch('http://localhost:3000/journal/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(newJournal),
            }); 
            const data = await res.json();
            set((state) => ({ journals: [...state.journals, data.data] }));
            return { success: true, message: "Journal entry created successfully." };

        },

        fetchJournals: async () => {
            const res = await fetch('http://localhost:3000/journal/getAll');
            const { success, data } = await res.json();
            if (success) {
                set({ journals: data });
            }
            return { success, journals: data };

        },

        deleteJournals: async (jid) => {
            const res = await fetch(`http://localhost:3000/journal/del/${jid}`,{
                method: 'DELETE'
            })
            const { success, message } = await res.json();

            if(!success) return { success: false, message:'Couldnt delete, error'}


            set((state) => ({ journals: state.journals.filter((journal) => journal._id !== jid)}))
            return {success: success , message : message}
        },

        updateJournal: async (pid, updatedJournal) => {
            const res = await fetch(`http://localhost:3000/journal/edit/${pid}`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(updatedJournal),
            });
            const data = await res.json();
            if (!data.success) return { success: false, message: data.message };

            // update the ui immediately, without needing a refresh
            set((state) => ({
                journals: state.journals.map((journal) => (journal._id === pid ? data.data : journal)),
            }));

            return { success: true, message: data.message };
        },

        



    }))

    export { useJournalStore }