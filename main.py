import sys
import os
import logging
import traceback
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QProgressBar, QMessageBox, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from pydub import AudioSegment
import tempfile
import whisper
import logging
import torch
from moviepy import VideoFileClip
from langchain.chains import LLMChain
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
logging.basicConfig(level=logging.DEBUG)

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели Whisper
try:
    logging.debug("Loading Whisper model...")
    model = whisper.load_model("turbo", device=device)
    logging.debug("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise


class TranscriptionThread(QThread):
    progress_signal = pyqtSignal(int)
    transcription_complete_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

    def run(self):
        try:
            logging.debug(f"Starting transcription for file: {self.file_path}")

            audio = self.extract_audio(self.file_path)
            if not audio:
                raise ValueError(f"Unable to load audio from file: {self.file_path}")

            segment_duration = 60 * 1000
            segments = self.split_audio(audio, segment_duration)
            total_steps = len(segments)

            transcription = ""

            for i, segment in enumerate(segments):
                if not self._is_running:
                    break

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    segment.export(temp_audio_file.name, format="wav")
                    temp_file_path = temp_audio_file.name

                text = self.transcribe_audio(temp_file_path)
                transcription += text + " "

                self.progress_signal.emit(int((i + 1) / total_steps * 100))

            self.save_transcription_to_file(transcription.strip())

            summarized_text = summarize_text(transcription.strip())
            output_file_path = self.get_output_file_path(self.file_path)
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(summarized_text)

            self.transcription_complete_signal.emit(output_file_path)
        except Exception as e:
            logging.error(f"Error in transcription thread: {e}")
            error_message = traceback.format_exc()
            self.error_signal.emit(f"Error: {error_message}")

    def extract_audio(self, file_path):
        try:
            if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.mpeg')):
                logging.debug(f"Extracting audio from video file: {file_path}")
                video = VideoFileClip(file_path)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    video.audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le")
                    return AudioSegment.from_file(temp_audio_file.name)
            else:
                logging.debug(f"Loading audio file: {file_path}")
                return AudioSegment.from_file(file_path)
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return None

    def stop(self):
        self._is_running = False
        self.wait()  # Wait for the thread to finish before continuing

    def split_audio(self, audio, segment_duration):
        return [audio[start:start + segment_duration] for start in range(0, len(audio), segment_duration)]

    def transcribe_audio(self, audio_file):
        try:
            logging.debug(f"Transcribing file: {audio_file}")
            result = model.transcribe(audio_file)
            return result.get('text', "Transcription error: Empty result")
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return f"Transcription error: {str(e)}"

    def get_output_file_path(self, input_file_path):
        directory, file_name = os.path.split(input_file_path)
        base_name, _ = os.path.splitext(file_name)
        return os.path.join(directory, f"{base_name}_summary.txt")

    def save_transcription_to_file(self, transcription_text):
        """
        Save the transcription text to a file for debugging.
        """
        directory, file_name = os.path.split(self.file_path)
        base_name, _ = os.path.splitext(file_name)
        text_file_path = os.path.join(directory, f"{base_name}_full_transcription.txt")

        with open(text_file_path, 'w', encoding='utf-8') as file:
            file.write(transcription_text)

        logging.debug(f"Full transcription saved to: {text_file_path}")


def initialize_llm():
    """
    Инициализация LLM для обработки текста.
    """
    try:
        template = "{question}"
        prompt = PromptTemplate.from_template(template)
        llm = YandexGPT(os.getenv(api_key), os.getenv(folder_id))
        return prompt | llm  # Используем новый формат вместо LLMChain
    except Exception as e:
        logging.error(f"Ошибка инициализации LLM: {e}")
        raise RuntimeError(f"Ошибка инициализации LLM: {e}")

def initialize_llm():
    """
    Инициализация LLM для обработки текста.
    """
    try:
        template = "{question}"
        prompt = PromptTemplate.from_template(template)
        llm = YandexGPT(os.getenv(api_key), os.getenv(folder_id))
        return LLMChain(prompt=prompt, llm=llm)
    except Exception as e:
        logging.error(f"Ошибка инициализации LLM: {e}")
        raise RuntimeError(f"Ошибка инициализации LLM: {e}")


def summarize_text(text):
    """
    Суммаризация текста, используя план и добавление заголовков.
    """
    try:
        # Log the text being passed to summarize_text
        logging.debug(f"Text to summarize: {text[:1000]}...")  # Print first 1000 chars for brevity

        llm_chain = initialize_llm()

        # Генерация краткого плана текста
        plan_prompt = f"Составь краткий план текста. Обозначь основные моменты - {text}"
        plan = llm_chain.invoke(plan_prompt)['text']

        # Генерация связного текста на основе плана
        text_prompt = f"{text} - На основе написанного текста, используя план - {plan}. Напиши складный текст"
        text_of_plan = llm_chain.invoke(text_prompt)['text']

        # Добавление заголовков к каждому абзацу
        headers_prompt = f"{text_of_plan} - Озаглавь данный текст. К каждому абзацу добавь мини заголовок"
        text_of_plan_headers = llm_chain.invoke(headers_prompt)['text']

        return text_of_plan_headers
    except Exception as e:
        logging.error(f"Ошибка суммаризации: {e}")
        raise RuntimeError(f"Ошибка суммаризации: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Summarizing the results of online conferences using AI methods")
        self.resize(600, 250)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 50, 300, 30)
        self.progress_bar.setVisible(False)

        self.transcription_thread = None

        self.init_ui()

    def init_ui(self):
        # Создаем центральный виджет и устанавливаем его
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Метка для названия проекта (по центру)
        self.title_label = QLabel("Подведение итогов онлайн-конференций с помощью методов ИИ", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Выравнивание по центру
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")  # Увеличенный шрифт и жирный текст

        # Метка для информации об авторах и школе (по правому краю)
        self.authors_label = QLabel(
            "ШКОЛА №1523 Г. МОСКВЫ\nКлянчин Михаил 10С\nЦветков Трофим 10Т",
            self
        )
        self.authors_label.setAlignment(Qt.AlignmentFlag.AlignRight)  # Выравнивание по правому краю
        self.authors_label.setStyleSheet("font-size: 12px;")  # Уменьшенный шрифт

        # Создаем кнопку для выбора файла
        self.select_file_button = QPushButton("Select Media File", self)
        self.select_file_button.clicked.connect(self.select_file)

        # Компоновка: вертикальная для всех элементов
        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.authors_label)
        layout.addSpacing(20)  # Добавляем отступ между текстом и кнопкой
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.progress_bar)
        layout.setContentsMargins(20, 20, 20, 20)  # Устанавливаем отступы вокруг элементов

        # Установка компоновки в центральный виджет
        central_widget.setLayout(layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Audio or a Video File", "", "Media Files (*.wav *.mp3 *.ogg *.mp4 *.mkv *.mpeg)")
        if file_path:
            logging.debug(f"Selected file: {file_path}")
            self.start_transcription(file_path)
        else:
            QMessageBox.warning(self, "Error", "No file selected.")

    def start_transcription(self, file_path):
        self.select_file_button.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.transcription_thread = TranscriptionThread(file_path)
        self.transcription_thread.progress_signal.connect(self.update_progress)
        self.transcription_thread.transcription_complete_signal.connect(self.transcription_complete)
        self.transcription_thread.error_signal.connect(self.handle_error)
        self.transcription_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def transcription_complete(self, output_file_path):
        logging.info(f"Transcription complete. Output saved at: {output_file_path}")
        QMessageBox.information(self, "Success", f"File saved at: {output_file_path}")
        print(f"Path to the output text: {output_file_path}")
        sys.exit()

    def handle_error(self, error_message):
        logging.error(f"Error during transcription: {error_message}")
        QMessageBox.critical(self, "Error", error_message)
        sys.exit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
