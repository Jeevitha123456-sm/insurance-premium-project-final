// script.js

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("premiumForm");
    const premiumOutput = document.getElementById("premiumOutput");
    const justificationOutput = document.getElementById("justificationOutput");
    const similarRecordsOutput = document.getElementById("similarRecordsOutput");
    const premiumValue = document.getElementById("premiumValue");
    const justificationValue = document.getElementById("justificationValue");
    const similarRecordsContainer = document.getElementById("similarRecordsContainer");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Get input values
        const name = document.getElementById("name").value.trim();
        const age = parseInt(document.getElementById("age").value);
        const sex = document.getElementById("sex").value;
        const region = document.getElementById("region").value.trim();
        const bmi = parseFloat(document.getElementById("bmi").value);
        const smoker = document.getElementById("smoker").checked;
        const sum_insured = parseFloat(document.getElementById("sum_insured").value);

        // Basic input validation
        if (!name || !age || !sex || !region || isNaN(bmi) || !sum_insured) {
            alert("Please fill in all required fields with valid values.");
            return;
        }

        // Show loading messages
        premiumOutput.style.display = "block";
        justificationOutput.style.display = "block";
        similarRecordsOutput.style.display = "block";
        premiumValue.textContent = "Calculating premium using Random Forest...";
        justificationValue.textContent = "Generating justification from LLM... Please wait.";
        similarRecordsContainer.innerHTML = '<p>Finding similar insurance profiles...</p>';

        try {
            const response = await fetch("/calculate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    risk_profile: { name, age, sex, region, bmi, smoker },
                    coverage: { sum_insured }
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Ensure justification is a string
                let justificationText = data.justification;

                // Handle Gemini nested response
                if (typeof justificationText === "object") {
                    if (justificationText.parts && justificationText.parts.length > 0) {
                        justificationText = justificationText.parts[0].text || JSON.stringify(justificationText);
                    } else if (justificationText.content) {
                        justificationText = justificationText.content;
                    } else {
                        justificationText = JSON.stringify(justificationText, null, 2);
                    }
                }

                // Display results
                const modelInfo = data.model_used ? `<div class="model-info">🤖 Prediction Model: <strong>${data.model_used === 'random_forest' ? 'Random Forest' : 'Fallback Formula'}</strong></div>` : '';
                premiumValue.innerHTML = `Estimated Premium: <strong>$${data.premium}</strong>${modelInfo}`;
                justificationValue.innerHTML = justificationText.replace(/\n/g, '<br>');
                
                // Display similar records if available
                if (data.similar_records && data.similar_records.length > 0) {
                    displaySimilarRecords(data.similar_records);
                } else {
                    similarRecordsContainer.innerHTML = '<p>No similar profiles found.</p>';
                }
                
                justificationOutput.scrollIntoView({ behavior: "smooth" });

            } else {
                premiumValue.textContent = "Error calculating premium.";
                justificationValue.textContent = `Reason: ${data.message || "Unknown error."}`;
                similarRecordsContainer.innerHTML = '';
            }

        } catch (error) {
            console.error("Error:", error);
            premiumValue.textContent = "Error calculating premium.";
            justificationValue.textContent = "Failed to generate justification. Check console.";
            similarRecordsContainer.innerHTML = '';
        }
    });

    function displaySimilarRecords(records) {
        similarRecordsContainer.innerHTML = '';
        records.forEach((record, index) => {
            const card = document.createElement('div');
            card.className = 'similar-card';
            card.innerHTML = `
                <p><strong>${index + 1}. Age ${record.age}, ${record.sex === 'male' ? 'Male' : 'Female'}, BMI ${record.bmi}</strong></p>
                <p>Region: ${record.region} | Children: ${record.children} | Smoker: ${record.smoker === 'yes' ? 'Yes' : 'No'}</p>
                <p><strong>Premium Coverage:</strong> $${record.charges}</p>
                <span class="similarity-score">Similarity: ${(record.similarity_score * 100).toFixed(1)}%</span>
            `;
            similarRecordsContainer.appendChild(card);
        });
    }

    async function loadHistory() {
        try {
            const response = await fetch("/get_history", {
                method: "GET"
            });

            const data = await response.json();

            if (response.ok && data.data && data.data.length > 0) {
                historyContainer.innerHTML = "";
                data.data.forEach(record => {
                    const card = document.createElement("div");
                    card.className = "history-card";
                    const date = new Date(record.created_at).toLocaleDateString() + " " + new Date(record.created_at).toLocaleTimeString();
                    card.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div style="flex: 1;">
                                <h4>${record.name} (Age: ${record.age})</h4>
                                <p><strong>Premium:</strong> $${record.premium}</p>
                                <p><strong>Sum Insured:</strong> $${record.sum_insured}</p>
                                <p><strong>Region:</strong> ${record.region} | <strong>BMI:</strong> ${record.bmi} | <strong>Smoker:</strong> ${record.smoker ? "Yes" : "No"}</p>
                                <p><strong>Date:</strong> ${date}</p>
                            </div>
                            <div style="display: flex; gap: 8px; margin-left: 10px;">
                                <button class="history-btn view-btn" data-record-id="${record.id}" title="View Details">View</button>
                                <button class="history-btn delete-btn" data-record-id="${record.id}" title="Delete">Delete</button>
                            </div>
                        </div>
                    `;
                    historyContainer.appendChild(card);
                    
                    // Add event listeners to buttons
                    const viewBtn = card.querySelector(".view-btn");
                    const deleteBtn = card.querySelector(".delete-btn");
                    
                    viewBtn.addEventListener("click", () => viewHistoryDetail(record));
                    deleteBtn.addEventListener("click", () => deleteHistoryRecord(record.id, card));
                });
            } else {
                historyContainer.innerHTML = '<div class="history-empty">No calculation history yet</div>';
            }
        } catch (error) {
            console.error("Error loading history:", error);
            historyContainer.innerHTML = '<div class="history-empty">Failed to load history</div>';
        }
    }

    function viewHistoryDetail(record) {
        const date = new Date(record.created_at).toLocaleDateString() + " " + new Date(record.created_at).toLocaleTimeString();
        const details = `
            <strong>Name:</strong> ${record.name}<br>
            <strong>Age:</strong> ${record.age}<br>
            <strong>Sex:</strong> ${record.sex}<br>
            <strong>Region:</strong> ${record.region}<br>
            <strong>BMI:</strong> ${record.bmi}<br>
            <strong>Smoker:</strong> ${record.smoker ? "Yes" : "No"}<br>
            <strong>Sum Insured:</strong> $${record.sum_insured}<br>
            <strong>Premium:</strong> $${record.premium}<br>
            <strong>Date:</strong> ${date}<br><br>
            <strong>Justification:</strong><br>
            ${record.justification.replace(/\n/g, '<br>')}
        `;
        alert(details.replace(/<br>/g, '\n'));
    }

    async function deleteHistoryRecord(recordId, cardElement) {
        if (confirm("Are you sure you want to delete this record?")) {
            try {
                const response = await fetch(`/delete_history/${recordId}`, {
                    method: "DELETE"
                });

                const data = await response.json();

                if (response.ok) {
                    cardElement.remove();
                    // Check if there are any history cards left
                    if (historyContainer.children.length === 0) {
                        historyContainer.innerHTML = '<div class="history-empty">No calculation history yet</div>';
                    }
                } else {
                    alert("Error deleting record: " + (data.message || "Unknown error"));
                }
            } catch (error) {
                console.error("Error deleting history:", error);
                alert("Failed to delete record");
            }
        }
    }
});
