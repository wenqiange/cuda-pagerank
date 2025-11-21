#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>

#include <sys/times.h>
#include <sys/resource.h>

int main() {
    std::ifstream file("enwiki-2013.txt");
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el fichero\n";
        return 1;
    }

    std::string line;
    std::unordered_map<int,int> outDegree;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        int from, to;
        std::stringstream ss(line);
        ss >> from >> to;
        outDegree[to]++;
    }

    file.close();

    std::cout << "To-degree(123) = " << outDegree[2055892] << std::endl;
    return 0;
}
