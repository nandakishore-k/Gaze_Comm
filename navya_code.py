from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QTextBrowser
from PyQt5.QtCore import Qt
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize navigation variables
        self.current_row = 0
        self.current_col = 0

        # OptiType keyboard layout
        self.keyboard_layout = [
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],  # Row 0
            ['J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'],  # Row 1
            ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']   # Row 2
        ]

        # UI Setup
        self.setWindowTitle("OptiType Keyboard")
        self.setGeometry(100, 100, 1200, 800)

        # Text Area for displaying typed text (Leave space at top)
        self.text_area = QTextBrowser(self)
        self.text_area.setGeometry(50, 150, 1100, 100)
        self.text_area.setText("Typed Text Here...")

        # Word Palette
        self.word_palette = []
        for i in range(4):
            label = QLabel(f"Word {i + 1}", self)
            label.setGeometry(50 + i * 275, 270, 250, 50)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: lightblue; border: 1px solid black;")
            self.word_palette.append(label)

        # Keyboard Buttons
        self.keyboard_buttons = []
        for i, row in enumerate(self.keyboard_layout):
            button_row = []
            for j, letter in enumerate(row):
                button = QPushButton(letter, self)
                button.setGeometry(100 + j * 110, 350 + i * 100, 100, 80)
                button.setStyleSheet("background-color: lightgray;")
                button_row.append(button)
            self.keyboard_buttons.append(button_row)

        # Highlight the first key
        self.update_highlight()

        # Navigation Buttons (Positioned horizontally at the top)
        button_width = 80
        button_height = 50
        button_x_start = 400
        button_y = 80  # Leave space at the top for navigation

        self.up_button = QPushButton("↑", self)
        self.up_button.setGeometry(button_x_start + 0 * (button_width + 10), button_y, button_width, button_height)
        self.up_button.clicked.connect(self.move_up)

        self.left_button = QPushButton("←", self)
        self.left_button.setGeometry(button_x_start + 1 * (button_width + 10), button_y, button_width, button_height)
        self.left_button.clicked.connect(self.move_left)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.setGeometry(button_x_start + 2 * (button_width + 10), button_y, button_width, button_height)
        self.ok_button.clicked.connect(self.select_key)

        self.right_button = QPushButton("→", self)
        self.right_button.setGeometry(button_x_start + 3 * (button_width + 10), button_y, button_width, button_height)
        self.right_button.clicked.connect(self.move_right)

        self.down_button = QPushButton("↓", self)
        self.down_button.setGeometry(button_x_start + 4 * (button_width + 10), button_y, button_width, button_height)
        self.down_button.clicked.connect(self.move_down)

    def update_highlight(self):
        """Highlight the current key and reset others."""
        for i, row in enumerate(self.keyboard_buttons):
            for j, button in enumerate(row):
                if i == self.current_row and j == self.current_col:
                    button.setStyleSheet("background-color: yellow;")  # Highlight
                else:
                    button.setStyleSheet("background-color: lightgray;")  # Default

    def move_up(self):
        """Move the highlight up."""
        if self.current_row > 0:
            self.current_row -= 1
        self.update_highlight()

    def move_down(self):
        """Move the highlight down."""
        if self.current_row < len(self.keyboard_layout) - 1:
            self.current_row += 1
        self.update_highlight()

    def move_left(self):
        """Move the highlight left."""
        if self.current_col > 0:
            self.current_col -= 1
        self.update_highlight()

    def move_right(self):
        """Move the highlight right."""
        if self.current_col < len(self.keyboard_layout[self.current_row]) - 1:
            self.current_col += 1
        self.update_highlight()

    def select_key(self):
        """Select the highlighted key and append it to the text area."""
        selected_key = self.keyboard_layout[self.current_row][self.current_col]
        current_text = self.text_area.toPlainText()
        if current_text == "Typed Text Here...":
            current_text = ""  # Clear the placeholder text
        self.text_area.setText(current_text + selected_key)


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
