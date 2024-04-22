import math
import os
import sys
import time

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidgetItem
from demodulator_gui import UI_Demodulator
from threading import Thread
import pyaudio
import numpy as np
import pyqtgraph as pg
from scipy.io import wavfile

from modem.QAMModem import QAMModem, ascii_to_string

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 48000
CHUNK = 512
RECORD_SECONDS = 10
THRESHOLD_START = 0.2  # Adjust this threshold according to your needs
THRESHOLD_STOP = 0.01


class AudioProcessingThread(Thread):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.stop_event = False

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)

    def run(self):
        print("Demodulator Running...")

        frames = []
        is_recording = False

        while not self.stop_event:
            data = self.stream.read(CHUNK)
            decoded_data = np.frombuffer(data, dtype=np.float32)
            amplitude = np.max(abs(decoded_data))

            if amplitude > THRESHOLD_START and not is_recording:
                print("Recording started.")
                is_recording = True

            if is_recording:
                frames.append(decoded_data)

            if amplitude < THRESHOLD_STOP and is_recording:
                print("Recording stopped.")

                wavfile.write(self.filename, RATE, np.concatenate(frames))
                print("Recording saved as", self.filename)
                frames.clear()
                is_recording = False

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def stop(self):
        self.stop_event = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = UI_Demodulator()
        self.ui.setupUi(self)

        # create modem
        self.modem = QAMModem(800)
        self.filename = "ReceivedAudio.wav"
        self.last_modified_time = os.path.getmtime(self.filename)
        self.dropped_packets = 0
        self.received_packets = 0

        # handle start button
        self.ui.btn_start.clicked.connect(self.start_recording)
        self.audio_thread = None

        # handle file change
        self.timer = QTimer()
        self.timer.timeout.connect(self.demodulate)
        self.timer.start(100)

        # handle symbol rate adjustment
        self.ui.txt_symbolrate.returnPressed.connect(self.on_symbol_rate_change)

        self.setup_graphs()
        self.setup_gui()

    def demodulate(self):
        """
        Checks to see if the audio file has changed since the last demodulation, if so, demodulate it and
        update channel stats.
        """
        try:
            current_modified_time = os.path.getmtime(self.filename)
            if current_modified_time != self.last_modified_time:
                self.received_packets = self.received_packets + 1
                self.last_modified_time = current_modified_time

                start = time.time()
                fs, rcc = wavfile.read(self.filename)
                assert fs == 48000

                rx_symbols = self.modem.demodulate_signal(rcc, plots=False)
                data = self.modem.get_data_from_stream(rx_symbols, plots=False)
                if data is None:
                    self.dropped_packets = self.dropped_packets + 1
                else:
                    rec_ascii = ''.join(char for char in ascii_to_string(data) if 0 < ord(char) < 128)
                    print("Recovered ASCII: ", rec_ascii)
                    print("Processing time: ", time.time() - start, " s")
                    self.ui.list_receivedmessages.addItem(QListWidgetItem(str(rec_ascii)))

        except Exception as e:
            print("Error:", e)

        self.ui.lbl_receivedpackets.setText(str(self.received_packets))
        self.ui.lbl_droppedpackets.setText(str(self.dropped_packets))

    def start_recording(self):
        if self.audio_thread and self.audio_thread.is_alive():
            print("Stopping Demodulator...")
            self.audio_thread.stop()
            self.audio_thread.join()
            self.ui.lbl_indicator.setStyleSheet("color: red;")
            return

        self.audio_thread = AudioProcessingThread(self.filename)
        self.audio_thread.start()
        self.ui.lbl_indicator.setStyleSheet("color: green;")

    def setup_graphs(self):
        self.ui.constellation_widget.setBackground('w')
        self.ui.constellation_widget.setXRange(-1, 1)
        self.ui.constellation_widget.setYRange(-1, 1)
        self.ui.constellation_widget.hideAxis('left')
        self.ui.constellation_widget.hideAxis('bottom')
        self.ui.constellation_widget.setTitle("Constellation Map (M=" + str(self.modem.M) + ")")

        ideal_x = np.array([point[0] for point in self.modem.constellation_map().values()])
        ideal_y = np.array([point[1] for point in self.modem.constellation_map().values()])

        ideal_x = ideal_x / np.max(ideal_x)
        ideal_y = ideal_y / np.max(ideal_y)

        self.ui.constellation_widget.addItem(pg.ScatterPlotItem(x=ideal_x, y=ideal_y))  # plot ideals

        # create plot object to live updating map
        constellation_plot = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 50), size=4)
        self.ui.constellation_widget.addItem(constellation_plot)
        self.modem.set_constellation_plot(constellation_plot)

    def setup_gui(self):
        self.ui.txt_symbolrate.setText("800")

        try:
            self.ui.lbl_effective_bitrate.setText(str(int(int(self.ui.txt_symbolrate.text()) * math.log2(self.modem.M))) + " bps")
        except Exception as e:
            print(e)

    def on_symbol_rate_change(self):
        try:
            self.ui.lbl_effective_bitrate.setText(str(int(int(self.ui.txt_symbolrate.text()) * math.log2(self.modem.M))) + " bps")
        except Exception as e:
            print(e)
            return

        self.modem.set_symbol_rate(int(self.ui.txt_symbolrate.text()))


def main():
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    # todo switch when moving to pi
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # else:
    #     if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    #         QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
