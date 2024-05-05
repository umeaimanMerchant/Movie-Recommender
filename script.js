let currentPart = 1;

function nextPart(part) {
    const currentPartDiv = document.getElementById(`part${part}`);
    const nextPartDiv = document.getElementById(`part${part + 1}`);

    if (currentPartDiv && nextPartDiv) {
        currentPartDiv.style.display = 'none';
        nextPartDiv.style.display = 'block';
        currentPart = part + 1;
    }
}


function submitForm() {
    const apiUrl = 'http://127.0.0.1:8000/model';

    // Define the user ratings for each genre
    const data = {
        action: document.getElementById('new_action').value,
        adventure: document.getElementById('new_adventure').value,
        animation: document.getElementById('new_animation').value,
        childrens: document.getElementById('new_childrens').value,
        comedy: document.getElementById('new_comedy').value,
        crime: document.getElementById('new_crime').value,
        documentary: document.getElementById('new_documentary').value,
        drama: document.getElementById('new_drama').value,
        fantasy: document.getElementById('new_fantasy').value,
        horror: document.getElementById('new_horror').value,
        mystery: document.getElementById('new_mystery').value,
        romance: document.getElementById('new_romance').value,
        scifi: document.getElementById('new_scifi').value,
        thriller: document.getElementById('new_thriller').value
    };

    const requestOptions = {
        method: 'POST',
        referrerPolicy: 'no-referrer',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    };
    

    fetch(apiUrl, requestOptions)
    //changes
        .then(response => response.json())
        .then(responseData => {
            // hide form 
            document.getElementById('part1').style.display = 'none';
            document.getElementById('part2').style.display = 'none';
            document.getElementById('part3').style.display = 'none';
            document.getElementById('submitButton').style.display = 'none';

            // Display response data as key-value pairs
            const responseContainer = document.getElementById('responseContainer');
            responseContainer.innerHTML = ''; // Clear previous content
            for (const key in responseData) {
                if (responseData.hasOwnProperty(key)) {
                    const keyValueElement = document.createElement('p');
                    keyValueElement.textContent = `${key}: ${responseData[key]}`;
                    responseContainer.appendChild(keyValueElement);
                }
            }
            document.getElementById('restartButton').style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
        });

}

function restartApp() {
    // Reset the form to its initial state
    document.getElementById('part1').style.display = 'block';
    document.getElementById('part2').style.display = 'none';
    document.getElementById('part3').style.display = 'none';

    // Show the submit button
    document.getElementById('submitButton').style.display = 'block';

    // Hide the restart button again
    document.getElementById('restartButton').style.display = 'none';

    // Reset the current part variable
    currentPart = 1;

    // Clear the response container
    document.getElementById('responseContainer').innerHTML = '';
}

