document.addEventListener('DOMContentLoaded', () => {
    const fetchBtn = document.getElementById('fetch-tasks');
    const actionsList = document.getElementById('actions-list');
    
    // Upload API Integration
    const fileInput = document.getElementById('csv-upload');
    const triggerUpload = document.getElementById('trigger-upload');
    const uploadStatus = document.getElementById('upload-status');
    
    triggerUpload.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        triggerUpload.disabled = true;
        triggerUpload.textContent = 'Agent Cleaning Initialized...';
        uploadStatus.textContent = 'Allocating Model Llama-3.1-8b-instant... Please Wait (up to 30s) ✨';
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `cleaned_${file.name}`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                
                uploadStatus.textContent = 'Success! Cleaned Dataset Downloaded 🚀';
                uploadStatus.style.color = 'var(--accent-3)';
            } else {
                const err = await response.text();
                throw new Error("Server processed invalid stream: " + err);
            }
        } catch (error) {
            console.error(error);
            uploadStatus.textContent = 'Agent Sequence Failed. Check Server Logs ❌';
            uploadStatus.style.color = '#ef4444';
        } finally {
            triggerUpload.disabled = false;
            triggerUpload.textContent = 'Select Another CSV File';
            fileInput.value = ''; // reset
        }
    });

    const generateSampleBtn = document.getElementById('generate-sample');
    generateSampleBtn.addEventListener('click', async () => {
        generateSampleBtn.disabled = true;
        generateSampleBtn.textContent = 'Synthesizing Data...';
        uploadStatus.textContent = 'Generating random unmapped chaos...';
        
        try {
            const response = await fetch('/generate_sample');
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                // Get filename from header if possible, else default
                let filename = 'judge_dummy_data.csv';
                const disposition = response.headers.get('Content-Disposition');
                if (disposition && disposition.indexOf('attachment') !== -1) {
                    const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                    if (matches != null && matches[1]) filename = matches[1].replace(/['"]/g, '');
                }
                
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                
                uploadStatus.textContent = 'Dummy Dataset Successfully Generated & Downloaded!';
                uploadStatus.style.color = 'var(--accent-3)';
            } else {
                throw new Error("Generator offline.");
            }
        } catch (err) {
            console.error(err);
            uploadStatus.textContent = 'Generation Failed ❌';
            uploadStatus.style.color = '#ef4444';
        } finally {
            generateSampleBtn.disabled = false;
            generateSampleBtn.textContent = 'Generate Dummy Data';
        }
    });

    loadData();

    async function loadData() {
        try {
            const response = await fetch('/tasks');
            if (response.ok) {
                const data = await response.json();
                actionsList.innerHTML = '';
                data.action_space.forEach(action => {
                    const pill = document.createElement('div');
                    pill.className = 'action-pill';
                    pill.textContent = action;
                    actionsList.appendChild(pill);
                });
            }
        } catch (error) {
            console.error("Failed to map environment endpoints.", error);
            actionsList.innerHTML = `<span style="color: #ef4444;">API endpoints offline. Boot Python server to sync.</span>`;
        }
    }
});
