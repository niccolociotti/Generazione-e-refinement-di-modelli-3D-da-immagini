using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.Networking;
using System.Collections;
using System.Text;
using System;

public class MRImageGenerator : MonoBehaviour
{
    [Header("UI Elements")]
    public TMP_InputField promptInputField; 
    public Button generateButton;           
    public RawImage outputImagePanel;       

    [Header("Server Settings")]
    public string pythonServerURL = "http://127.0.0.1:5000/generate-image"; 

    void Start()
    {
        generateButton.onClick.AddListener(OnGenerateButtonClicked);
    }

    void OnGenerateButtonClicked()
    {
        string promptText = promptInputField.text;
        
        if (!string.IsNullOrEmpty(promptText))
        {
            generateButton.interactable = false; 
            StartCoroutine(RequestImageGeneration(promptText));
        }
    }

    IEnumerator RequestImageGeneration(string prompt)
    {
        string jsonPayload = $"{{\"prompt\": \"{prompt}\"}}";
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);

        Debug.Log("Fase 1: Inviando richiesta di generazione al server...");

        using (UnityWebRequest request = new UnityWebRequest(pythonServerURL, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Risposta JSON ricevuta!");
                StartCoroutine(ProcessServerResponse(request.downloadHandler.text));
            }
            else
            {
                Debug.LogError("Errore dal server: " + request.error);
                generateButton.interactable = true;
            }
        }
    }

    IEnumerator ProcessServerResponse(string jsonResponse)
    {
        string imageUrlToDownload = null;

        // 1. Leggiamo i dati in modo sicuro
        try
        {
            ServerResponse responseData = JsonUtility.FromJson<ServerResponse>(jsonResponse);

            if (responseData != null && responseData.status == "ok" && !string.IsNullOrEmpty(responseData.image_url))
            {
                imageUrlToDownload = responseData.image_url;
            }
            else
            {
                Debug.LogError("Il server ha restituito un errore o nessun URL valido.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Errore nel parsing del JSON dal server: " + e.Message);
        }

        // 2. Scarichiamo l'immagine FUORI dal blocco try-catch (che risolve l'errore CS1626)
        if (!string.IsNullOrEmpty(imageUrlToDownload))
        {
            Debug.Log("Fase 2: Scaricando l'immagine dall'URL: " + imageUrlToDownload);
            
            using (UnityWebRequest textureRequest = UnityWebRequestTexture.GetTexture(imageUrlToDownload))
            {
                yield return textureRequest.SendWebRequest();

                if (textureRequest.result == UnityWebRequest.Result.Success)
                {
                    Texture2D downloadedTexture = DownloadHandlerTexture.GetContent(textureRequest);
                    outputImagePanel.texture = downloadedTexture;
                    
                    if(outputImagePanel.GetComponent<AspectRatioFitter>() != null)
                        outputImagePanel.GetComponent<AspectRatioFitter>().aspectRatio = (float)downloadedTexture.width / downloadedTexture.height;
                        
                    Debug.Log("Immagine applicata con successo in Mixed Reality!");
                }
                else
                {
                    Debug.LogError("Errore nel download dell'immagine: " + textureRequest.error);
                }
            }
        }

        // Riattiviamo il bottone
        generateButton.interactable = true;
    }

    [Serializable]
    private class ServerResponse
    {
        public string status;
        public string image_path;
        public string image_url; 
    }
}