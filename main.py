print("Initializing UI")

import sys, os
import numpy as np
import pandas as pd
import math
import h5py  # ADDED: for loading HDF5 localization files (e.g. from Picasso)

from PyQt6.QtWidgets import QDialog, QPushButton, QApplication, QLabel, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QWidget, QVBoxLayout, QListWidgetItem, QDockWidget, QStatusBar, QProgressBar, QHBoxLayout, QLineEdit, QInputDialog

from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal, QMetaObject
from PyQt6 import QtCore
from PyQt6.QtGui import QGuiApplication, QDoubleValidator
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from MixtureModelAlgorithm import EM1, EM2, EM3  # Import from the original script
from BlinkExtractionAlgorithm import Cluster2d1d
from LocalPrecisionAlgorithm import Loc_Acc
from PyVistaPlotter import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

os.environ["TRAITSUI_TOOLKIT"] = "qt"
os.environ["ETS_TOOLKIT"] = "qt"

# ADDED: deterministic color for a given cluster ID — consistent across all plots
def _cluster_color(cluster_id):
    golden_ratio = 0.618033988749895
    hue = (cluster_id * golden_ratio) % 1.0
    h = hue * 6
    c, m = 0.8, 0.2
    x = c * (1 - abs(h % 2 - 1))
    if   h < 1: r, g, b = c, x, 0
    elif h < 2: r, g, b = x, c, 0
    elif h < 3: r, g, b = 0, c, x
    elif h < 4: r, g, b = 0, x, c
    elif h < 5: r, g, b = x, 0, c
    else:        r, g, b = c, 0, x
    return (r + m, g + m, b + m)


class AboutDialog(QDialog):
    def __init__(self):
        super(AboutDialog, self).__init__()

        self.setWindowTitle("About")

        # Layout
        layout = QVBoxLayout()

        # Program information
        info_label = QLabel(
            "Protein Stoichiometry Quantifier\n\n"
            "Date: 2024-11\n"
            "Developed by: Eric Shi in the Milstein Lab, University of Toronto\n"
            "This program utilizes algorithms developed by:\n"
            "Artittaya Boonkird, Daniel F Nino and Joshua N Milstein in the Milstein Lab: https://doi.org/10.1093/bioadv/vbab032 for the prediction of protein stoichiometry\n"
            "Ulrike Endesfelder, Sebastian Malkusch, Franziska Fricke and Mike Heilemann: https://pubmed.ncbi.nlm.nih.gov/24522395/ for the estimation of localization precision\n"
        )
        info_label.setOpenExternalLinks(True)  # Allow clickable links
        info_label.setWordWrap(True)

        layout.addWidget(info_label)

        # OK button to close the dialog
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


class ModifyAttributesDialog(QDialog):
    def __init__(self, parent=None):
        super(ModifyAttributesDialog, self).__init__(parent)
        self.setWindowTitle("Modify Default Attributes")
        
        # Layout
        layout = QVBoxLayout()
        
        # Create input field for max_iter
        max_iter_layout = QHBoxLayout()
        max_iter_label = QLabel("Maximum Iterations:")
        self.max_iter_input = QLineEdit()
        self.max_iter_input.setText("50000")  # Default value
        max_iter_layout.addWidget(max_iter_label)
        max_iter_layout.addWidget(self.max_iter_input)
        layout.addLayout(max_iter_layout)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Add OK and Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def get_max_iter(self):
        return int(self.max_iter_input.text())


class DataHandler:
    def __init__(self, main_window):
        self.main_window = main_window  # Store a reference to the MainWindow
        self.blinking_data = None
        self.blinking_data_imported = False
        self.localization_data = None
        self.localization_data_imported = False
        self.local_precision = -1
        self.local_precision_error = None  # Store any error message

    def load_blinking(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self.main_window, "Open Data File", "", "CSV Files (*.csv)")

        if file_path:
            self.main_window.blinkingFilePathLabel.setText(f"Loaded Blinking Dataset: {file_path}")
            try:
                self.blinking_data = np.genfromtxt(file_path, delimiter=",")
                self.blinking_data_imported = True
                self.main_window.blinking_data = self.blinking_data 
                self.main_window.blinking_data_imported = self.blinking_data_imported
                self.main_window.stoichiometry_clicked(None)  # Switch tabs
            except Exception as e:
                self.main_window.blinkingFilePathLabel.setText(f"Error loading file: {e}")
                self.blinking_data_imported = False  # Set to False if loading fails
                self.blinking_data = None
                self.main_window.blinking_data = self.blinking_data # Update MainWindow variable
                self.main_window.blinking_data_imported = self.blinking_data_imported

        elif self.blinking_data_imported: # Keep the previous data if the user cancels the file dialog and data was already loaded
             pass
        else:
            self.main_window.blinkingFilePathLabel.setText("No file loaded")


    def load_localization(self):
        file_dialog = QFileDialog()
        # ADDED: accept HDF5 files in addition to the original .txt format
        file_path, _ = file_dialog.getOpenFileName(
            self.main_window, "Open Data File", "",
            "Localization Files (*.txt *.hdf5 *.h5);;Text Files (*.txt);;HDF5 Files (*.hdf5 *.h5)"
        )

        if file_path:
            try:
                # ADDED: branch for HDF5 files (e.g. exported from Picasso)
                if file_path.lower().endswith(('.hdf5', '.h5')):
                    with h5py.File(file_path, 'r') as f:
                        if 'locs' not in f:
                            raise ValueError("HDF5 file does not contain a 'locs' dataset. Expected Picasso format.")
                        locs = f['locs'][:]
                        required = {'x', 'y', 'frame', 'photons'}
                        available = set(locs.dtype.names) if locs.dtype.names else set()
                        missing = required - available
                        if missing:
                            raise ValueError(f"HDF5 file is missing required columns: {missing}")
                        # ADDED: ask user for pixel size to convert pixel coordinates to nanometers
                        pixel_size, ok = QInputDialog.getDouble(
                            self.main_window, "Pixel Size",
                            "Enter camera pixel size (nm/pixel):\n"
                            "(Coordinates will be converted from pixels to nm\n"
                            "so the DBSCAN epsilon parameter works correctly.)",
                            130.0, 1.0, 10000.0, 1
                        )
                        if not ok:
                            return
                        # Build DataFrame with columns in the order the app expects: x, y, frame, intensity
                        self.localization_data = pd.DataFrame({
                            'x':       locs['x'].astype(float) * pixel_size,
                            'y':       locs['y'].astype(float) * pixel_size,
                            'frame':   locs['frame'].astype(int),
                            'photons': locs['photons'].astype(float),
                        })
                else:
                    # Original .txt loading path (unchanged)
                    with open(file_path, 'r') as f:
                        header = f.readline()
                        first_line = f.readline().strip()
                        if not first_line:
                            raise ValueError("File is empty after header")
                        columns = first_line.split()
                        if len(columns) < 4:
                            raise ValueError("File must have at least 4 columns")
                        try:
                            float(columns[0])
                            float(columns[1])
                            int(columns[2])
                            float(columns[3])
                        except ValueError as e:
                            raise ValueError("Invalid data types in columns. Expected: float/int, float/int, int, float/int")
                    self.localization_data = pd.read_csv(file_path, delimiter=' ', header=1)

                self.localization_data_imported = True
                self.main_window.localization_data = self.localization_data 
                self.main_window.localization_data_imported = self.localization_data_imported
                self.main_window.localizationFilePathLabel.setText(f"Loaded Localization Dataset: {file_path}")
                # Enable the Extract Blinks button
                self.main_window.runExtractionButton.setEnabled(True)
                # Disable the Proceed button until extraction is done
                self.main_window.pushButton.setEnabled(False)

                try:
                    p, e = Loc_Acc(self.localization_data)
                    self.local_precision = p
                    self.local_precision_error = e
                    self.main_window.local_precision = self.local_precision

                    item = QListWidgetItem(f"{p:.2f}±({e:.2f})")
                    self.main_window.valueListWidget.insertItem(1, item)
                    self.main_window.preprocessing_clicked(None)  # Switch tabs
                except Exception as exc:
                    item = QListWidgetItem(f"Optimal parameter not found")
                    self.main_window.valueListWidget.insertItem(1, item)
                    self.main_window.preprocessing_clicked(None)  # Switch tabs

            except Exception as ex:
                self.main_window.show_popup("Invalid File Format", 
                    "The file must be space-separated with the following format:\n"
                    "- First column (x-position): float or integer\n"
                    "- Second column (y-position): float or integer\n"
                    "- Third column (frame number): integer\n"
                    "- Fourth column (intensity): float or integer\n"
                    f"\nError: {str(ex)}")
                self.localization_data_imported = False
                self.localization_data = None
                self.local_precision = -1
                self.local_precision_error = None
                self.main_window.localization_data = self.localization_data
                self.main_window.localization_data_imported = self.localization_data_imported
                self.main_window.local_precision = self.local_precision
                self.main_window.localizationFilePathLabel.setText("No file loaded")
                # Disable both buttons when data is invalid
                self.main_window.runExtractionButton.setEnabled(False)
                self.main_window.pushButton.setEnabled(False)

        elif self.localization_data_imported:  # Keep the previous data if the user cancels the file dialog and data was already loaded
            pass
        else:
            self.main_window.localizationFilePathLabel.setText("No file loaded")
            self.local_precision = -1  # Reset if no file is loaded
            self.local_precision_error = None
            self.main_window.local_precision = self.local_precision
            # Disable both buttons when no data is loaded
            self.main_window.runExtractionButton.setEnabled(False)
            self.main_window.pushButton.setEnabled(False)


class EMAlgorithmExecution(QThread):
    finished_signal = pyqtSignal(object, object, object, object, object, object)  # Signal for results
    progress_update = pyqtSignal(int)  # Signal for progress updates
    cancelled_signal = pyqtSignal()  # Signal for cancellation
    max_iter_error_signal = pyqtSignal(str, str)  # Signal for max iterations error (title, message)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.lab_ineff = False
        self.model = None
        self.is_cancelled = False  # Flag to track cancellation
        # Connect the max_iter_error_signal to the show_popup method
        self.max_iter_error_signal.connect(self.main_window.show_popup)

    def run(self):  # Override the run method (this is what the thread executes)
        theta_input = self.main_window.inputTheta.text()

        self.main_window.replicates = int(self.main_window.replicatesInput.text())
        self.main_window.subset_factor = float(self.main_window.subsetSizeInput.text())

        theta = float(theta_input) if theta_input else None

        self.lab_ineff = True if theta else False

        m, d, t = "M", "M/D", "M/D/T"

        if self.main_window.radioEM1.isChecked():
            self.model = m
        elif self.main_window.radioEM2.isChecked():
            self.model = d
        else:
            self.model = t

        pi_replicates, lam_replicates, aic_replicates = self._get_replicates(self.model, theta) # Pass self.model

        # If we got empty results (max iterations exceeded), emit empty results
        if not pi_replicates or not lam_replicates or not aic_replicates:
            self.finished_signal.emit(None, None, None, None, None, self.model)
            return

        transposed_pi = list(zip(*pi_replicates))
        transposed_lam = lam_replicates
        transposed_aic = aic_replicates

        pi_means = [np.mean(sublist) for sublist in transposed_pi]
        pi_stds = [np.std(sublist) for sublist in transposed_pi]

        lam_means = np.mean(transposed_lam)
        aic_means = np.mean(transposed_aic)
        lam_std = np.std(transposed_lam)

        # Emit the signal with the results:
        self.finished_signal.emit(lam_means, pi_means, aic_means, pi_stds, lam_std, self.model)

    def _get_replicates(self, model, theta):
        bootstrapped_data = self._bootstrap_dataset(self.main_window.replicates, self.main_window.subset_factor)
        pi_replicates = []
        lam_replicates = []
        aic_replicates = []
        progress = 0

        for i, dataset in enumerate(bootstrapped_data):
            if self.is_cancelled:  # Check cancellation flag in the loop
                self.cancelled_signal.emit()  # Emit cancellation signal
                return [], [], []  # Return empty lists to stop further processing
            try:
                if model == "M":
                    em1 = EM1(dataset)
                    em1.initialize()
                    em1.run(max_iter=self.main_window.max_iter)
                    pi_replicates.append(em1.pi)
                    lam_replicates.append(em1.lam)
                    aic_replicates.append(em1.AIC)
                elif model == "M/D":
                    em2 = EM2(dataset)
                    em2.initialize()
                    em2.run(max_iter=self.main_window.max_iter)
                    if self.lab_ineff:
                        em2.theta = theta
                        em2.apply_lab_ineff()
                    pi_replicates.append(em2.pi)
                    lam_replicates.append(em2.lam)
                    aic_replicates.append(em2.AIC)
                else:  # model == "M/D/T"
                    em3 = EM3(dataset)
                    em3.initialize()
                    em3.run(max_iter=self.main_window.max_iter)
                    if self.lab_ineff:
                        em3.theta = theta
                        em3.apply_lab_ineff()
                    pi_replicates.append(em3.pi)
                    lam_replicates.append(em3.lam)
                    aic_replicates.append(em3.AIC)
            except RuntimeError as e:
                if "Maximum iterations" in str(e):
                    self.max_iter_error_signal.emit(
                        "Maximum Iterations Exceeded",
                        f"The EM algorithm exceeded the maximum number of iterations ({self.main_window.max_iter}). This can be modified in the settings menu."
                    )
                    return [], [], []  # Return empty lists to stop further processing

            progress_percentage = int(round((i + 1) / len(bootstrapped_data) * 100))
            self.progress_update.emit(progress_percentage)  # Emit the progress update signal

        return pi_replicates, lam_replicates, aic_replicates

    def _bootstrap_dataset(self, replicates, size_fraction):
        return [np.random.choice(self.main_window.blinking_data, size=math.floor(len(self.main_window.blinking_data) * size_fraction), replace=False) for _ in range(replicates)]

    def cancel(self):  # Method to set the cancellation flag
        self.is_cancelled = True

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        ui_path = self.resource_path("main.ui")

        uic.loadUi(ui_path, self)  # Load the UI file

        positive_validator = QDoubleValidator()
        positive_validator.setRange(0.0, float('inf'))
        positive_validator.setDecimals(3)

        self.initialize_connections()

        self.inputTheta.setText("1")
        self.inputTheta.setValidator(positive_validator)

        self.preprocessing_clicked(None)

        # Initially disable buttons
        self.runExtractionButton.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.graphButton.setEnabled(False)
        self.graph2dButton.setEnabled(False)
        self.runEMButton.setEnabled(False)

        self.blinking_data = None
        self.blinking_data_imported = False
        self.localization_data = None
        self.localization_data_imported = False
        self.lab_ineff = False
        self.analyzer = None

        self.replicates = 1
        self.subset_factor = 1
        self.local_precision = -1
        self.max_iter = 50000  # Default value for max iterations

        self.initialize_stoichiometry_graph()
        self.initialize_blinking_graph()
        self.set_window_size()
        
        self.data_handler = DataHandler(self)  # Pass 'self' (the MainWindow instance)
        self.em_thread = EMAlgorithmExecution(self)
        self.em_thread.finished_signal.connect(self.handle_em_results)
        self.em_thread.started.connect(self.thread_started)
        self.em_thread.finished.connect(self.thread_finished)
        self.em_thread.progress_update.connect(self.update_progress_bar)
        self.em_thread.cancelled_signal.connect(self.algorithm_cancelled)


        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.progressBar = QProgressBar() 
        self.statusBar.addPermanentWidget(self.progressBar)  # Add it to the status bar
        self.progressBar.setMaximumHeight(10)  # Set maximum height to make it more compact
        self.progressBar.hide()

        self.cancel_button = QPushButton("Cancel", self.statusBar)
        self.statusBar.addPermanentWidget(self.cancel_button)
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_em_algorithm)

    def show_about_dialog(self):
        """Display the About dialog."""
        about_dialog = AboutDialog()
        about_dialog.exec()  # Show the dialog modally
    
    def set_window_size(self):
        # Get the primary screen's geometry
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Calculate the new dimensions (2/3 of the screen size)
        new_width = int(screen_width * 2 / 3)
        new_height = int(screen_height * 2 / 3)

        # Center the window and resize
        self.setGeometry(
            int((screen_width - new_width) / 2),
            int((screen_height - new_height) / 2),
            new_width,
            new_height,
        )

    def initialize_connections(self):
        # Connect the buttons to their respective functions
        self.runEMButton.clicked.connect(self.run_replicates)
        self.runExtractionButton.clicked.connect(self.run_blink_extraction)
        self.graphButton.clicked.connect(self.choose_graph)
        self.pushButton.clicked.connect(self.proceed_to_stoichiometry)  # Add connection for proceed button

        # Connect Menu items to their functions
        self.actionLoadBlinking.triggered.connect(self.load_blinking)
        self.actionLoadLocalization.triggered.connect(self.load_localization)
        self.actionGraph_Dataset.triggered.connect(self.plot_dataset)
        self.actionAbout.triggered.connect(self.show_about_dialog)
        self.actionModify_Attributes.triggered.connect(self.show_modify_attributes_dialog)

        # ADDED: Export Cluster Data menu item (added programmatically to the File menu)
        self.menuFile.addSeparator()
        self.actionExportClusterData = self.menuFile.addAction("Export Cluster Data")
        self.actionExportClusterData.triggered.connect(self.export_cluster_data)

        self.graph2dButton.clicked.connect(self.graph_2d_gaussian)

        self.preprocessingSwitch.mousePressEvent = self.preprocessing_clicked
        self.stoichiometrySwitch.mousePressEvent = self.stoichiometry_clicked

    def initialize_stoichiometry_graph(self):
        plt.rc('font', family='Arial')  # FIXED: Calibri is Windows-only, Arial is available on macOS

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        self.bars = self.ax.bar(range(3), [0, 0, 0], color='gray', edgecolor='black', width=0.5)        # Set the X-axis labels and Y-axis limits
        self.ax.set_xticks(range(3))
        self.ax.set_xticklabels(['Monomer', 'Dimer', 'Trimer'])
        self.ax.set_ylim(0, 1)  # Set the y-axis limits to 0-100
        self.ax.set_ylabel("Distribution")
        self.ax.tick_params(axis='both', labelsize=9)

        # Create a canvas and embed it in the graphWidget
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.graphWidget.layout().addWidget(self.canvas)

    def initialize_blinking_graph(self):

        self.fig2, (self.ax2, self.ax3) = plt.subplots(1, 2, figsize=(14, 5))

        self.ax2.set_xlabel("Cluster ID")
        self.ax2.set_ylabel("Number of Blinks")
        self.ax2.tick_params(axis='both', labelsize=9)

        self.ax3.set_xlabel("Number of Blinks")
        self.ax3.set_ylabel("Frequency")
        self.ax3.tick_params(axis='both', labelsize=9)

        self.fig2.tight_layout()
        self.canvas2 = FigureCanvasQTAgg(self.fig2)
        self.blinkGraph.layout().addWidget(self.canvas2)

        self._blink_bars = None
        self._blink_bar_data = []
        self._blink_annotation = None
        self.canvas2.mpl_connect('motion_notify_event', self._on_blink_hover)

    def _on_blink_hover(self, event):
        if event.inaxes != self.ax2 or self._blink_bars is None:
            if self._blink_annotation:
                self._blink_annotation.set_visible(False)
                self.canvas2.draw_idle()
            return

        for i, bar in enumerate(self._blink_bars):
            if bar.contains(event)[0] and i < len(self._blink_bar_data):
                cid, count = self._blink_bar_data[i]
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                label = f'Cluster {cid}\n{count} blinks'
                if self._blink_annotation is None:
                    self._blink_annotation = self.ax2.annotate(
                        label, xy=(x, y), xytext=(0, 8), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
                    )
                else:
                    self._blink_annotation.set_text(label)
                    self._blink_annotation.xy = (x, y)
                    self._blink_annotation.set_visible(True)
                self.canvas2.draw_idle()
                return

        if self._blink_annotation:
            self._blink_annotation.set_visible(False)
            self.canvas2.draw_idle()

    def resource_path(self, relative_path):
        """ Get the absolute path to the resource, works in development and after PyInstaller packaging """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def preprocessing_clicked(self, event):
        self.preprocessingSwitch.setStyleSheet("color: black;")
        self.stoichiometrySwitch.setStyleSheet("color: gray;")
        self.confStack.setCurrentIndex(1)
        self.resultStack.setCurrentIndex(1)
        self.graphStack.setCurrentIndex(1)

    def stoichiometry_clicked(self, event):
        self.preprocessingSwitch.setStyleSheet("color: gray;")
        self.stoichiometrySwitch.setStyleSheet("color: black;")
        self.confStack.setCurrentIndex(0)
        self.resultStack.setCurrentIndex(0)
        self.graphStack.setCurrentIndex(0)

    def load_blinking(self):
        self.data_handler.load_blinking()
        # Enable the EM button if data was loaded successfully
        self.runEMButton.setEnabled(self.blinking_data_imported)

    def load_localization(self):
        self.data_handler.load_localization()
        # Enable/disable buttons based on data state
        self.runExtractionButton.setEnabled(self.localization_data_imported)
        # Both graph buttons should be disabled until extraction is done
        self.graphButton.setEnabled(False)
        self.graph2dButton.setEnabled(False)
        self.pushButton.setEnabled(False)  # Disabled until extraction is done

    def run_blink_extraction(self):
        if not self.localization_data_imported:
            self.show_popup("Missing data", "Please load a localization file before running the extraction")
            return

        self.analyzer = Cluster2d1d(self.localization_data)
        self.analyzer.epsilon = int(self.epsInput.text())
        self.analyzer.min_sample = int(self.minSampleInput.text())
        self.analyzer.proximity = int(self.proxInput.text())
        self.analyzer.extract_features()
        self.analyzer.perform_dbscan()
        self.analyzer.get_all_temporal_clusters()
        blinking_data = self.analyzer.get_blinking_data()
        blinking_with_ids = self.analyzer.get_blinking_data_with_ids()
        self.display_blinking_data(blinking_with_ids)
        self.plot_blinking(blinking_with_ids)  # MODIFIED: pass ID-paired data for color coding

        # ADDED: show DBSCAN cluster statistics after extraction
        stats = self.analyzer.get_cluster_stats()
        self.show_popup(
            "Extraction Complete",
            f"Total localizations:     {stats['total_localizations']}\n"
            f"Kept (in clusters):      {stats['kept']}\n"
            f"Filtered out (noise):    {stats['noise_filtered']}\n"
            f"Clusters found:          {stats['n_clusters']}"
        )

        # Enable all buttons after successful extraction
        self.pushButton.setEnabled(True)
        self.graphButton.setEnabled(True)
        self.graph2dButton.setEnabled(True)

    def choose_graph(self):
        if self.radioOriginal.isChecked():
            if self.localization_data_imported:
                update_plot_pyvista(self.localization_data)
        elif self.radioSpatial.isChecked():
            if self.analyzer:
                # MODIFIED: pass cluster IDs for consistent color coding
                visualize_spatial_clusters_pyvista(self.analyzer.all_temporal_clusters, self.localization_data,
                                                   cluster_ids=self.analyzer.clusters_2d)
        else:
            if self.analyzer:
                # MODIFIED: pass cluster IDs for consistent color coding
                visualize_temporal_clusters_pyvista(self.analyzer.all_temporal_clusters, self.localization_data,
                                                    cluster_ids=self.analyzer.clusters_2d)

    # MODIFIED: now accepts (cluster_id, blink_count) pairs and displays both
    def display_blinking_data(self, data_with_ids):
        text_to_display = "\n".join(f"Cluster {cid}: {count} blinks" for cid, count in data_with_ids)
        self.blinkListDisplay.setText(text_to_display)

    def display_results(self, lam, pi, AIC, pi_std, lam_std, model):
        
        if not all(0 <= value <= 1 for value in pi if value is not None):
            QMessageBox.warning(self, "Unphysical Values Predicted", "Predicted distribution values are outside the expected range (0-1). This may indicate an issue with the data or chosen parameters.")

        row_position = 0
        self.tableWidget.insertRow(row_position)

        # Add Result Values to the new row
        item = QTableWidgetItem(str(round(pi[0]*100, 1)))
        # Make the item non-editable
        # item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.tableWidget.setItem(row_position, 0, item)

        if model == "M":
            item = QTableWidgetItem(str("N/A"))            
        else:
            item = QTableWidgetItem(str(round(pi[1]*100, 1)))
        self.tableWidget.setItem(row_position, 1, item)

        if model != "M/D/T":
            item = QTableWidgetItem(str("N/A"))
        else:
            item = QTableWidgetItem(str(round(pi[2]*100, 1)))
        self.tableWidget.setItem(row_position, 2, item)

        item = QTableWidgetItem(str(round(lam, 2)))
        self.tableWidget.setItem(row_position, 3, item)

        item = QTableWidgetItem(str(round(AIC, 2)))
        self.tableWidget.setItem(row_position, 4, item)

        item = QTableWidgetItem(model)
        self.tableWidget.setItem(row_position, 5, item)

    def plot_stoichiometry(self, values, std, model):
        """
        Plots the given values as a bar graph.

        Args:
            values (list): A list of three values to plot.
        """

        self.ax.clear()

        self.bars = self.ax.bar(range(3), [0, 0, 0], color='gray', edgecolor='black', width=0.5)        # Set the X-axis labels and Y-axis limits
        self.ax.set_xticks(range(3))
        self.ax.set_xticklabels(['Monomer', 'Dimer', 'Trimer'])
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Distribution")
        self.ax.tick_params(axis='both', labelsize=9)

        for bar, value in zip(self.bars, values):
            bar.set_height(value)

        self.ax.errorbar(range(len(std)), values, yerr=std, fmt='none', 
                     ecolor='black', capsize=5, capthick=2)

        self.canvas.draw()

    def show_popup(self, title, message):
        QMessageBox.information(self, title, message)

    # MODIFIED: now accepts (cluster_id, blink_count) pairs and colors each bar by cluster ID
    def plot_blinking(self, blinking_with_ids):
        self.ax2.clear()
        self.ax3.clear()
        self._blink_annotation = None  # reset annotation when data changes

        cluster_ids = [cid for cid, _ in blinking_with_ids]
        counts = [count for _, count in blinking_with_ids]
        colors = [_cluster_color(cid) for cid in cluster_ids]

        self._blink_bars = self.ax2.bar(range(len(cluster_ids)), counts, color=colors, edgecolor='gray', linewidth=0.5)
        self._blink_bar_data = list(blinking_with_ids)
        self.ax2.set_xticks([])
        self.ax2.set_xlabel("Cluster ID")
        self.ax2.set_ylabel("Number of Blinks")

        self.ax3.hist(counts, bins='auto', color='steelblue', edgecolor='white')
        self.ax3.set_xlabel("Number of Blinks")
        self.ax3.set_ylabel("Frequency")

        self.fig2.tight_layout()
        self.canvas2.draw()

    def plot_dataset(self):
        
        if not self.blinking_data_imported:
            self.show_popup("Missing data", "Please import the data file before plotting")
            return

        self.ax.clear()

        self.ax.plot(sorted(self.blinking_data))
        self.ax.set_xlabel("Dye (Sorted)")
        self.ax.set_ylabel("Number of Blinks")

        self.canvas.draw()

    def graph_2d_gaussian(self):
        if self.analyzer:
            max_res = 8192
            alpha_scale = 0.8
            if self.radio2dOriginal.isChecked():
                if self.local_precision <= 0:
                    self.show_popup("Missing Precision",
                        "Localization precision could not be estimated from this dataset.\n"
                        "The Gaussian render requires a valid precision value.\n\n"
                        "Try the 2D Points view instead.")
                    return
                self.analyzer.plot_original_gaussian(self.local_precision, alpha_scale=alpha_scale, intensity_scale=0.3, min_alpha=0.05, max_res=max_res)
            elif self.radio2dClusters.isChecked():
                if self.local_precision <= 0:
                    self.show_popup("Missing Precision",
                        "Localization precision could not be estimated from this dataset.\n"
                        "The Gaussian render requires a valid precision value.\n\n"
                        "Try the 2D Points view instead.")
                    return
                self.analyzer.plot_gaussian_clusters(self.local_precision, alpha_scale=alpha_scale, intensity_scale=0.3, min_alpha=0.05, max_res=max_res)
            elif self.radio2dPoints.isChecked():
                if not self.analyzer.all_temporal_clusters:
                    self.show_popup("No Data", "Please run the extraction algorithm first.")
                    return
                # MODIFIED: pass cluster IDs for correct labeling and color coding
                plot_2d_points_clusters(self.analyzer.all_temporal_clusters, self.localization_data,
                                        cluster_ids=self.analyzer.clusters_2d)

    def run_replicates(self):
        if self.blinking_data is None:
            self.show_popup("Missing data", "Please load a data file before running the algorithm")
            return
        
        m, d, t = "M", "M/D", "M/D/T"

        if self.radioEM1.isChecked():
            self.model = m
        elif self.radioEM2.isChecked():
            self.model = d
        elif self.radioEM3.isChecked():
            self.model = t
        else:
            self.show_popup("Missing algorithm", "Please select an algorithm")
            return
        self.em_thread.start()
        self.progressBar.show()
        self.progressBar.setValue(0)
        self.cancel_button.show()

    def thread_started(self):
        self.runEMButton.setEnabled(False)

    def thread_finished(self):
        self.runEMButton.setEnabled(True)
        self.progressBar.hide()
        self.cancel_button.hide()

    def update_progress_bar(self, progress):
        self.progressBar.setValue(progress)

    def handle_em_results(self, lam_means, pi_means, aic_means, pi_stds, lam_std, model):
        if self.em_thread.is_cancelled:
            self.em_thread.is_cancelled = False
            return
            
        # Check if we have empty results (which happens when max iterations is exceeded)
        if not pi_means or not pi_stds:
            return  # Just return without trying to display or plot results
            
        self.display_results(lam_means, pi_means, aic_means, pi_stds, lam_std, model)
        self.plot_stoichiometry(pi_means, pi_stds, model)
        self.runEMButton.setEnabled(True)

    def algorithm_cancelled(self):
        self.runEMButton.setEnabled(True)
        self.progressBar.hide()
        self.cancel_button.hide()
        self.show_popup("Algorithm Cancelled", "The EM algorithm has been cancelled.")

    def cancel_em_algorithm(self):
        self.em_thread.cancel()

    def show_modify_attributes_dialog(self):
        """Display the Modify Attributes dialog."""
        dialog = ModifyAttributesDialog(self)
        dialog.max_iter_input.setText(str(self.max_iter))  # Set current value
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.max_iter = dialog.get_max_iter()
            self.show_popup("Settings Updated", f"Maximum iterations has been set to {self.max_iter}")

    # ADDED: export localization data with cluster IDs to a CSV file
    def export_cluster_data(self):
        """Save a CSV containing every localization with its DBSCAN cluster_id.
        cluster_id = -1 means the localization was classified as noise and excluded."""
        if not self.analyzer or self.analyzer.clusters_2d is None:
            self.show_popup("No Data", "Please run the extraction algorithm first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Cluster Data", "cluster_data.csv", "CSV Files (*.csv)")
        if file_path:
            df = self.analyzer.get_localization_cluster_data()
            df.to_csv(file_path, index=False)
            self.show_popup("Export Complete", f"Cluster data saved to:\n{file_path}")

    def proceed_to_stoichiometry(self):
        """Switch to stoichiometry tab and transfer the extracted blink data."""
        if not self.analyzer or not self.analyzer.all_temporal_clusters:
            self.show_popup("No Data", "Please run the extraction algorithm first.")
            return
            
        # Get the blinking data
        blinking_data = self.analyzer.get_blinking_data()

        # ADDED: warn if any cluster has >= 1000 blinks (likely an outlier)
        blinking_with_ids = self.analyzer.get_blinking_data_with_ids()
        outliers = [(cid, count) for cid, count in blinking_with_ids if count >= 1000]
        if outliers:
            outlier_lines = "\n".join(f"  Cluster {cid}: {count} blinks" for cid, count in outliers)
            reply = QMessageBox.question(
                self, "Outlier Warning",
                f"The following clusters have ≥1000 blinks and may be outliers:\n\n{outlier_lines}\n\nProceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Load the data into the blinking_data variable
        self.blinking_data = np.array(blinking_data)
        self.blinking_data_imported = True

        # Switch to stoichiometry tab
        self.stoichiometry_clicked(None)

        # Update the file path label
        self.blinkingFilePathLabel.setText("Loaded Blinking Dataset: (from extraction)")
        
        # Enable the EM button since we now have data
        self.runEMButton.setEnabled(True)
        
        # Show a success message
        self.show_popup("Data Transferred", "Blink data has been successfully transferred to stoichiometry analysis.")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ADDED: force light palette so hardcoded light colors in the UI look correct on macOS dark mode
    from PyQt6.QtGui import QPalette, QColor
    app.setStyle("Fusion")
    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
    light_palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    light_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(light_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
