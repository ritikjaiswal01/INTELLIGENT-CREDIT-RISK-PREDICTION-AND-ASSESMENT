document.getElementById("predictionForm").addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = {
        age: parseInt(document.getElementById("age").value),
        income: parseFloat(document.getElementById("income").value),
        credit_score: parseInt(document.getElementById("credit_score").value),
        loan_amount: parseFloat(document.getElementById("loan_amount").value),
        debt_to_income: parseFloat(document.getElementById("debt_to_income").value),
        existing_loan: parseInt(document.getElementById("existing_loan").value),
    };

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.prediction) {
                document.getElementById("result").innerText = `Prediction: ${data.prediction}`;
            } else if (data.error) {
                document.getElementById("result").innerText = `Error: ${data.error}`;
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            document.getElementById("result").innerText = "An error occurred.";
        });
});
