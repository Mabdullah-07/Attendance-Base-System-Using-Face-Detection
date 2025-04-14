#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iomanip>

#ifdef _WIN32
#include <conio.h>  
#endif

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

static void clearScreen() {
#ifdef  _WIN32
    system("cls");
#endif
}

static void returnToMenu() {
    cout << "\nPress any key to return to the menu...";
#ifdef _WIN32
    _getch();
#endif
}

class AttendanceSystem {
public:
    AttendanceSystem() {
        // Create directories with full paths
        if (!fs::exists("E:\\Me stuff\\attendance_photos")) {
            fs::create_directory("E:\\Me stuff\\attendance_photos");
        }
        if (!fs::exists("E:\\Me stuff\\attendance_logs")) {
            fs::create_directory("E:\\Me stuff\\attendance_logs");
        }
    }

    void markStudentAttendance(const string& name, const string& rollNumber);
    void markTeacherAttendance(const string& name, const string& regNumber, int shiftNumber);
    void viewStudentAttendance() const;
    void viewTeacherAttendance() const;

private:
    const string PHOTOS_DIR = "E:\\Me stuff\\attendance_photos";
    const string LOGS_DIR = "E:\\Me stuff\\attendance_logs";
    const int REQUIRED_DETECTIONS = 30;
    const int TIMEOUT_SECONDS = 30;
    const string WINDOW_NAME = "Face Detection System";

    void saveToCSV(const string& name, const string& id, const string& photoPath, const string& csvFileName) const {
        string fullPath = LOGS_DIR + csvFileName + ".csv";
        bool fileExists = fs::exists(fullPath);

        ofstream csv(fullPath, ios::app);
        if (!csv.is_open()) {
            cerr << "Error: Could not open CSV file: " << fullPath << endl;
            return;
        }

        if (!fileExists) {
            csv << "Date,Time,Name,ID,Photo Path\n";
        }

        auto now = chrono::system_clock::now();
        auto nowTime = chrono::system_clock::to_time_t(now);
        tm localTime;
        localtime_s(&localTime, &nowTime);

        csv << put_time(&localTime, "%Y-%m-%d,")
            << put_time(&localTime, "%H:%M:%S,")
            << name << ","
            << id << ","
            << photoPath << endl;

        csv.close();

        if (!fs::exists(fullPath)) {
            cerr << "Error: CSV file was not created: " << fullPath << endl;
        }
    }

    void markAttendance(const string& name, const string& id, const string& logFileName) const {
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Error: Cannot open camera" << endl;
            return;
        }

        String cascadePath = "E:\\Me stuff\\projectOOP\\Project Codes\\FaceAttendanceSystem\\models\\haarcascade_frontalface_default.xml";
        CascadeClassifier faceDetector;
        if (!faceDetector.load(cascadePath)) {
            cerr << "Error: Cannot load face cascade classifier from: " << cascadePath << endl;
            return;
        }

        time_t startTime = time(0);
        bool faceDetected = false;
        int successfulDetections = 0;

        while (time(0) - startTime < TIMEOUT_SECONDS && !faceDetected) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);

            vector<Rect> faces;
            faceDetector.detectMultiScale(gray, faces, 1.1, 2, 0, Size(60, 60));

            rectangle(frame, Point(frame.cols / 4, frame.rows / 4),
                Point(3 * frame.cols / 4, 3 * frame.rows / 4),
                Scalar(255, 255, 255), 2);

            if (!faces.empty()) {
                Rect face = faces[0];
                if (face.x > frame.cols / 4 && face.x + face.width < 3 * frame.cols / 4 &&
                    face.y > frame.rows / 4 && face.y + face.height < 3 * frame.rows / 4) {

                    Rect expandedFace = face;
                    int expandX = face.width * 0.5;
                    int expandY = face.height * 0.5;
                    expandedFace.x = max(0, face.x - expandX / 2);
                    expandedFace.y = max(0, face.y - expandY / 2);
                    expandedFace.width = min(frame.cols - expandedFace.x, face.width + expandX);
                    expandedFace.height = min(frame.rows - expandedFace.y, face.height + expandY);

                    rectangle(frame, expandedFace, Scalar(0, 255, 0), 2);
                    successfulDetections++;

                    if (successfulDetections >= REQUIRED_DETECTIONS && !faceDetected) {
                        string timestamp = to_string(time(0));
                        string photoPath = PHOTOS_DIR + name + "_" + id + "_" + timestamp + ".jpg";

                        Mat detectedFace = frame(expandedFace);
                        bool saved = imwrite(photoPath, detectedFace);

                        if (saved) {
                            cout << "Face image saved: " << photoPath << endl;

                            saveToCSV(name, id, photoPath, logFileName);

                            faceDetected = true;
                            cout << "Attendance marked successfully!" << endl;
                        }
                        else {
                            cerr << "Error: Could not save face image to: " << photoPath << endl;
                        }
                    }
                }
                else {
                    rectangle(frame, face, Scalar(0, 165, 255), 2);
                    successfulDetections = 0;
                }
            }

            string status = faceDetected ? "Face detected and saved!" : "Position your face in the box";
            int timeLeft = TIMEOUT_SECONDS - (time(0) - startTime);

            putText(frame, status + " (Time left: " + to_string(timeLeft) + "s)",
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 2);

            int barWidth = frame.cols - 20;
            int progress = (successfulDetections * barWidth) / REQUIRED_DETECTIONS;
            rectangle(frame, Point(10, frame.rows - 40),
                Point(10 + barWidth, frame.rows - 20),
                Scalar(255, 255, 255), 1);
            rectangle(frame, Point(10, frame.rows - 40),
                Point(10 + progress, frame.rows - 20),
                Scalar(0, 255, 0), FILLED);

            imshow(WINDOW_NAME, frame);

            if (waitKey(1) == 27) break;
        }

        destroyAllWindows();
    }

    void displayAttendance(const string& logFileName) const {
        string csvPath = LOGS_DIR + logFileName + ".csv";
        ifstream csv(csvPath);

        if (!csv.is_open()) {
            cout << "No attendance records found at: " << csvPath << endl;
            return;
        }

        string line;
        cout << "\n=== Attendance Records ===\n\n";
        while (getline(csv, line)) {
            cout << line << endl;
        }
    }
};

void AttendanceSystem::markStudentAttendance(const string& name, const string& rollNumber) {
    markAttendance(name, rollNumber, "student_attendance");
    clearScreen();
    cout << "Student attendance marked for: " << name << " (" << rollNumber << ")" << endl;
    returnToMenu();
}

void AttendanceSystem::markTeacherAttendance(const string& name, const string& regNumber, int shiftNumber) {
    markAttendance(name, regNumber + "_shift" + to_string(shiftNumber), "teacher_attendance");
    clearScreen();
    cout << "Teacher attendance marked for: " << name << " (" << regNumber << ") Shift: " << shiftNumber << endl;
    returnToMenu();
}

void AttendanceSystem::viewStudentAttendance() const {
    clearScreen();
    displayAttendance("student_attendance");
    returnToMenu();
}

void AttendanceSystem::viewTeacherAttendance() const {
    clearScreen();
    displayAttendance("teacher_attendance");
    returnToMenu();
}

int main() {
    AttendanceSystem attendanceSystem;
    int choice;

    do {
        clearScreen();
        cout << "\n\t=== Face Recognition Attendance System ===\n\n";
        cout << "\t1) Mark Student Attendance\n";
        cout << "\t2) Mark Teacher Attendance\n";
        cout << "\t3) View Student Attendance\n";
        cout << "\t4) View Teacher Attendance\n";
        cout << "\t5) Exit\n";
        cout << "\n\tEnter your choice: ";
        cin >> choice;

        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        switch (choice) {
        case 1: {
            string name, rollNumber;
            cout << "\n\tEnter Student Name: ";
            getline(cin, name);
            cout << "\tEnter Roll Number: ";
            getline(cin, rollNumber);
            attendanceSystem.markStudentAttendance(name, rollNumber);
            break;
        }
        case 2: {
            string name, regNumber;
            int shiftNumber;
            cout << "\n\tEnter Teacher Name: ";
            getline(cin, name);
            cout << "\tEnter Registration Number: ";
            getline(cin, regNumber);
            cout << "\tEnter Shift Number: ";
            cin >> shiftNumber;
            attendanceSystem.markTeacherAttendance(name, regNumber, shiftNumber);
            break;
        }
        case 3:
            attendanceSystem.viewStudentAttendance();
            break;
        case 4:
            attendanceSystem.viewTeacherAttendance();
            break;
        case 5:
            cout << "\n\tExiting program. Goodbye!\n";
            break;
        default:
            cout << "\n\tInvalid choice. Please try again.\n";
            returnToMenu();
        }
    } while (choice != 5);

    return 0;
}