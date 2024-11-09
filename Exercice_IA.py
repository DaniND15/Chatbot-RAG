

import os
import streamlit as st
import spacy
from typing import List, Set, Dict, Any
from mistralai import Mistral, UserMessage, AssistantMessage, SystemMessage
from datetime import datetime
import json
import requests
from dotenv import load_dotenv


load_dotenv()
# Configuration de la page
st.set_page_config(
    page_title="Chatbot Mistral",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Mistral
# MODEL = "mistral-large-latest"
MODEL = "mistral-large-2407"
MISTRAL_API_KEY =  os.getenv("MISTRAL_API_KEY")
LYCEE_API_URL =  os.getenv("LYCEE_API_URL")


class ChatManager:
    def __init__(self, storage_file: str = "chat_history.json", archive_dir: str = "archived_chats"):
        self.storage_file = storage_file
        self.archive_dir = archive_dir
        os.makedirs(archive_dir, exist_ok=True)
        self.load_chats()

    def load_chats(self):
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.chats = json.load(f)
            else:
                self.chats = {}
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'historique: {e}")
            self.chats = {}

    def save_chats(self):
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.chats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde de l'historique: {e}")

    def create_new_chat(self, chat_name: str = None) -> str:
        chat_name = chat_name or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        chat_id = str(len(self.chats) + 1)
        self.chats[chat_id] = {
            "name": chat_name,
            "messages": [],
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "is_archived": False
        }
        self.save_chats()
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str) -> bool:
        if chat_id in self.chats and not self.chats[chat_id]["is_archived"]:
            self.chats[chat_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            self.save_chats()
            return True
        return False

    def rename_chat(self, chat_id: str, new_name: str) -> bool:
        if chat_id in self.chats:
            self.chats[chat_id]["name"] = new_name
            self.save_chats()
            return True
        return False

# class ChatNameGenerator:
#     def __init__(self):
#         try:
#             self.nlp = spacy.load()
#         except OSError:
#             st.error("Modèle spaCy manquant. Installez-le avec : python -m spacy download fr_core_news_sm")
#             self.nlp = None
            

#     def generate_chat_name(self, user_input: str, max_length: int = 40) -> str:
#         if not self.nlp:
#             return user_input[:max_length]
        
#         doc = self.nlp(user_input.strip())
#         custom_stops = {
#             "bonjour", "salut", "hello", "hey", "s'il vous plait", "svp", 
#             "pouvez-vous", "peux-tu", "j'aimerais", "je voudrais", "je veux"
#         }
        
#         key_phrases = [chunk.text for chunk in doc.noun_chunks 
#                       if chunk.text.lower() not in custom_stops]
        
#         chat_name = key_phrases[0] if key_phrases else user_input[:max_length]
#         return chat_name.capitalize()[:max_length]

def load_json_data() -> None:
    """Charge les données JSON dans la session Streamlit"""
    # if "json_data" not in st.session_state:
    try:
        # response = requests.get(LYCEE_API_URL, timeout=50)
        response = requests.get(LYCEE_API_URL)
        response.raise_for_status()
        text_response = response.text
        
        try:
            data = json.loads(text_response)
            if isinstance(data, dict) and "results" in data:
                st.session_state.json_data = data["results"]
            else:
                st.error("Structure JSON inattendue")
                st.session_state.json_data = []
        except json.JSONDecodeError as e:
            st.error(f"Erreur de décodage JSON: {e}")
            st.error(f"Début de la réponse: {text_response[:500]}")
            st.session_state.json_data = []
            
    except requests.RequestException as e:
        st.error(f"Erreur lors de la requête HTTP: {e}")
        st.session_state.json_data = []
    except Exception as e:
        st.error(f"Erreur inattendue: {e}")
        st.session_state.json_data = []

def format_context_for_mistral(json_data: List[Dict[str, Any]]) -> str:
    """Formate les données JSON pour le contexte de Mistral"""
    if not json_data:
        return ""
    
    try:
        context = "Voici les données disponibles sur les établissements scolaires:\n\n"
        
        for item in json_data:  # Limite à 5 établissements
            try:
                context += f"- {item.get('patronyme', 'Non spécifié')} ({item.get('annee_scolaire', 'Non spécifié')}):\n"
                context += f"  * Diplôme: {item.get('lib_diplome', 'Non spécifié')}\n"
                context += f"  * Effectif total: {item.get('effectif_total', 'Non spécifié')}\n"
                context += f"  * Catégorie: {item.get('lib_categorie', 'Non spécifié')}\n"
                context += "\n"
            except Exception as e:
                continue
                
        return context
    except Exception as e:
        st.error(f"Erreur lors du formatage du contexte: {e}")
        return ""

def get_mistral_response(client: Mistral, messages: List[Dict[str, str]]) -> str:
    """Obtient une réponse de l'API Mistral"""
    api_messages = []
    for msg in messages:
        if msg["role"] == "user":
            api_messages.append(UserMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            api_messages.append(AssistantMessage(content=msg["content"]))
        elif msg["role"] == "system":
            api_messages.append(SystemMessage(content=msg["content"]))

    response_container = st.empty()
    full_response = ""
    
    try:
        stream_response = client.chat.stream(model=MODEL, messages=api_messages)
        for chunk in stream_response:
            chunk_content = chunk.data.choices[0].delta.content
            full_response += chunk_content if chunk_content is not None else ""
            response_container.markdown(full_response)
        return full_response
    except Exception as e:
        error_message = f"Erreur lors de la communication avec Mistral : {e}"
        st.error(error_message)
        return error_message

def display_chat_history(chat_manager: ChatManager, chat_id: str) -> None:
    """Affiche l'historique des messages"""
    if chat_id in chat_manager.chats:
        for message in chat_manager.chats[chat_id]["messages"]:
            role = "Vous" if message["role"] == "user" else "Assistant"
            with st.chat_message(message["role"]):
                st.write(message["content"])
            st.markdown(f"<small>{message['timestamp']}</small>", unsafe_allow_html=True)

def prepare_context_and_messages(current_chat: Dict[str, Any]) -> List[Dict[str, str]]:
    """Prépare le contexte et les messages pour Mistral"""
    try:
        messages = current_chat["messages"].copy()
        
        if hasattr(st.session_state, 'json_data') and st.session_state.json_data:
            context = format_context_for_mistral(st.session_state.json_data)
            if context:
                system_message = (
                    "Vous êtes un expert dans l'analyse des données. "
                    "Voici le contexte des données disponibles sur les établissements:\n\n"
                    f"{context}\n"
                   "Utilisez uniquement les informations fournies pour répondre directement aux questions concernant les établissements scolaires. Ne commentez pas votre raisonnement. Si vous ne disposez pas de toutes les informations nécessaires, vous pouvez répondre de manière partielle, mais ne donnez jamais de fausse réponse. Assurez-vous que votre réponse respecte toujours les exigences de la question."
                )
                messages.append({
                    "role": "system",
                    "content": system_message
                })
        
        return messages
    except Exception as e:
        st.error(f"Erreur lors de la préparation du contexte: {e}")
        return current_chat["messages"].copy()

def main():
    if not MISTRAL_API_KEY:
        st.error("Clé API Mistral manquante. Veuillez configurer la variable d'environnement.")
        return

    # Initialisation des composants
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    # Chargement des données JSON
    load_json_data()

    # Interface utilisateur
    with st.sidebar:
        st.title("Conversations")
        
        # Affichage des conversations existantes
        for chat_id, chat in st.session_state.chat_manager.chats.items():
            if st.button(f"{chat['name']} ({chat_id})", key=f"chat_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        # Création d'une nouvelle conversation
        with st.expander("Nouvelle conversation", expanded=False):
            chat_name = st.text_input("Nom de la conversation", "Nouvelle conversation")
            if st.button("Créer"):
                chat_id = st.session_state.chat_manager.create_new_chat(chat_name)
                st.session_state.current_chat_id = chat_id
                st.rerun()

    # Affichage de la conversation courante
    if ("current_chat_id" in st.session_state and 
        st.session_state.current_chat_id in st.session_state.chat_manager.chats):
        current_chat = st.session_state.chat_manager.chats[st.session_state.current_chat_id]
        st.title(f"💬 Assistant IA ")
        
        
        with st.expander("Informations", expanded=False):
            st.write(f"Créée le : {current_chat['created_at']}")
            st.write(f"Statut : {'Archivée' if current_chat.get('is_archived', False) else 'Active'}")
    else:
        st.title("Chatbot Mistral")
        if "current_chat_id" not in st.session_state:
            new_chat_id = st.session_state.chat_manager.create_new_chat("Nouvelle conversation")
            st.session_state.current_chat_id = new_chat_id
            st.rerun()

    # Initialisation du client Mistral
    client = Mistral(api_key=MISTRAL_API_KEY)

    # Affichage de l'historique
    display_chat_history(st.session_state.chat_manager, st.session_state.current_chat_id)

    # Gestion des messages
    current_chat = st.session_state.chat_manager.chats[st.session_state.current_chat_id]
    if not current_chat.get("is_archived", False):
        user_input = st.chat_input("Votre message:")
        if user_input:
            # Affichage du message utilisateur
            st.chat_message("user").write(user_input)
            
            # # Génération du nom de la conversation si première interaction
            # if (len(current_chat["messages"]) == 0 and 
            #     current_chat["name"] == "Nouvelle conversation"):
            #     name_generator = ChatNameGenerator()
            #     chat_name = name_generator.generate_chat_name(user_input)
            #     st.session_state.chat_manager.rename_chat(
            #         st.session_state.current_chat_id, 
            #         chat_name
            #     )
            

            # Ajout du message utilisateur
            st.session_state.chat_manager.add_message(
                st.session_state.current_chat_id,
                "user",
                user_input
            )

            # Préparation et envoi des messages à Mistral
            messages = prepare_context_and_messages(current_chat)
            response = get_mistral_response(client, messages)
            
            # Ajout de la réponse à l'historique
            st.session_state.chat_manager.add_message(
                st.session_state.current_chat_id,
                "assistant",
                response
            )

if __name__ == "__main__":
    main()