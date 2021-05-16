// #include <sstream>
// #include "cc.cuh"

// int main(int argc, char** argv) {
//     cudaFree(0);
//     ArgumentParser arguments(argc, argv, true);
//     if (arguments.input.empty()) {
//         arguments.input = testGraphPath;
//     }
//     if (arguments.sourceNode == 0) {
//         arguments.sourceNode = 47235513;
//     }
//     //arguments.method =2;
//     if (arguments.method == 0) {
//         conventionParticipateCC(arguments.input);
//     } else if (arguments.method == 1){
//         ccShareTrace(arguments.input);
//     } else if (arguments.method == 2){
//         ccOpt(arguments.input, arguments.adviseK);
//     }
//     return 0;
// }

#include "CalculateOpt.cuh"
#include "main.cuh"
#include "constants.cuh"
#include "ArgumentParser.cuh"
int main(int argc, char** argv) {
    cudaFree(0);
    ArgumentParser arguments(argc, argv, true);
    if (arguments.algo.empty()) {
        arguments.algo = "bfs";
    }
    cout << "arguments.algo " << arguments.algo << endl;
    if (arguments.algo == "bfs") {
        bfs_opt(arguments.input, arguments.sourceNode, arguments.adviseK);
    } else if (arguments.algo == "cc") {
        cc_opt(arguments.input, arguments.adviseK);
    } else if (arguments.algo == "sssp") {
        sssp_opt(arguments.input, arguments.sourceNode, arguments.adviseK);
    } else if (arguments.algo == "pr") {
        pr_opt(arguments.input, arguments.adviseK);
    }
    
    return 0;
}
// #include "pagerank.cuh"
// int main(int argc, char** argv) {
//     //cout << "111111111111" << endl;
//     cudaFree(0);
//     ArgumentParser arguments(argc, argv, true);
//     if (arguments.input.empty()) {
//         arguments.input = prGraphPath;
//     }
//     if (arguments.sourceNode == 0) {
//         arguments.sourceNode = 47235513;

//     }
//     arguments.method = 0;
//     if (arguments.method == 0) {
//         conventionParticipatePR(arguments.input);
//     } else if (arguments.method == 1){
//         prShareByInDegreeTrace(arguments.input);
//     } else if (arguments.method == 2){
//         //arguments.adviseK = 0.000001;
//         prOpt(arguments.input, arguments.adviseK);
//         //conventionParticipatePR(arguments.input);
//         //conventionParticipatePRHalfStatic();
//     }
//     return 0;
// }
// #include "sssp.cuh"
// int main(int argc, char** argv) {
//     cudaFree(0);
//     ArgumentParser arguments(argc, argv, true);
//     if (arguments.input.empty()) {
//         arguments.input = ssspGraphPath;
//     }
//     if (arguments.sourceNode == 0) {
//         arguments.sourceNode = 25838548;
//     }
//     arguments.method = 0;
//     if (arguments.method == 0) {
//         conventionParticipateSSSP(arguments.sourceNode, arguments.input);
//     } else if (arguments.method == 1){
//         ssspShareTrace(arguments.sourceNode, arguments.input);
//     } else if (arguments.method == 2){
//         //arguments.adviseK = 0.8;
//         ssspOpt(arguments.sourceNode, arguments.input, arguments.adviseK);
//     }
//     return 0;
// }
// #include "bfs.cuh"
// int main(int argc, char** argv) {
//     cudaFree(0);
//     ArgumentParser arguments(argc, argv, true);
//     if (arguments.input.empty()) {
//         arguments.input = ssspGraphPath;
//     }
//     if (arguments.sourceNode == 0) {
//         arguments.sourceNode = 25838548;
//     }
//     arguments.method = 0;
//     if (arguments.method == 0) {
//         conventionParticipateBFS(arguments.input, arguments.sourceNode);
//     } else if (arguments.method == 1){
//         //ssspShareTrace(arguments.sourceNode, arguments.input);
//     } else if (arguments.method == 2){
//         //arguments.adviseK = 0.8;
//         //ssspOpt(arguments.sourceNode, arguments.input, arguments.adviseK);
//     }
//     return 0;
// }
/*#include "pagerank.cuh"

struct TestRecord {
    float startTime = 0.0f;
    long long startAddress = 0;
    long size = 0;
};

struct RecordByUnit {
    vector<float> touchList;
};

int main(int argc, char **argv) {
    //cout << "111111111111" << endl;
    cudaFree(0);
    ifstream infile;
    infile.open("./footprintlog-sssp-friendster.txt");
    stringstream ss;
    string line;
    long lineIndex = 0;
    vector<TestRecord> recordList;
    long min = LONG_MAX;
    long max = 0;
    while (getline(infile, line)) {
        ss.str("");
        ss.clear();
        ss << line;
        if (lineIndex > 5) {
            string startTimeStr = "";
            string startAddress = "";
            string pageSize = "";
            for (int i = 0; i < 19; i++) {
                string param;
                ss >> param;
                if (i == 0) {
                    startTimeStr = param;
                }
                if (i == 16) {
                    pageSize = param;
                }
                if (i == 17) {
                    startAddress = param;
                }
                if (i == 18) {
                    string type;
                    type += param;
                    ss >> param;
                    type += param;
                    ss >> param;
                    type += param;
                    ss >> param;
                    type += param;
                    if (type != "[UnifiedMemoryMemcpyHtoD]") {
                        break;
                        //cout << "line: " << lineIndex << " " << type << endl;
                    } else {
                        TestRecord record;
                        string temp = startTimeStr.substr(startTimeStr.size() - 2, startTimeStr.size());
                        if (temp == "ms") {
                            record.startTime = atof(startTimeStr.substr(0, startTimeStr.size() - 2).c_str()) / 1000;
                        } else {
                            record.startTime = atof(startTimeStr.substr(0, startTimeStr.size() - 1).c_str());
                        }
                        char *addressTemp;
                        record.startAddress = strtol(startAddress.substr(0, startAddress.size()).c_str(), &addressTemp,
                                                     16);
                        if (record.startAddress > max) {
                            max = record.startAddress;
                        }
                        if (record.startAddress < min) {
                            if (record.startAddress == 0) {
                                break;
                            }
                            if (record.startAddress == 4096) {
                                cout << "" << endl;
                            }
                            min = record.startAddress;
                        }
                        record.size = atoi(pageSize.substr(0, pageSize.size() - 2).c_str()) * 1024;
                        recordList.push_back(record);
                    }

                }
            }

        }

        lineIndex++;
    }
    infile.close();
    for (int i = 0; i < recordList.size(); i++) {
        recordList[i].startAddress -= min;
    }
    long unit = 1 << 24;
    vector<RecordByUnit> recorder((max - min) / unit + 1);
    for (int i = 0; i < recordList.size(); i++) {
        int recorderIndex;
        recorderIndex = recordList[i].startAddress / unit;
        if (recorder[recorderIndex].touchList.empty()) {
            recorder[recorderIndex].touchList.push_back(recordList[i].startTime);
        } else {
            float nowTouch = recorder[recorderIndex].touchList[recorder[recorderIndex].touchList.size() - 1];
            if (nowTouch > recordList[i].startTime) {
                recorder[recorderIndex].touchList[recorder[recorderIndex].touchList.size() - 1] = recordList[i].startTime;
            } else if (recordList[i].startTime - nowTouch > 1){
                recorder[recorderIndex].touchList.push_back(recordList[i].startTime);
            }
        }
    }
    ofstream fout("uvmFootprintFriendsterSSSP.txt");
    for (int i = 0; i < recorder.size(); i++) {
        fout << "recorder index " << i << " ";
        for (int j = 0; j < recorder[i].touchList.size(); j++) {
            fout << recorder[i].touchList[j] << " ";
        }
        fout << endl;
    }
    fout.close();
    return 0;
}*/
