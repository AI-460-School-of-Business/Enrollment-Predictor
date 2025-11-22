import { useState } from "react";
import { FileUpload } from "./components/FileUpload";
import { SemesterSelector } from "./components/SemesterSelector";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Checkbox } from "./components/ui/checkbox";
import { Download, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";

interface ResultData {
  id: string;
  confidenceLevel: number;
  seatsNeeded: number;
  courseNumber: string;
  courseTitle: string;
}

type SortColumn =
  | "confidenceLevel"
  | "seatsNeeded"
  | "courseNumber"
  | "courseTitle";
type SortDirection = "asc" | "desc";

// For Docker, set VITE_API_URL in env; for local dev, fallback to localhost
const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL ?? "http://localhost:5000";

export default function App() {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [model, setModel] = useState<string>("");
  const [selectedSemesters, setSelectedSemesters] = useState<string[]>([]);
  const [crnFilter, setCrnFilter] = useState("");
  const [departmentFilter, setDepartmentFilter] = useState<string>("");
  const [results, setResults] = useState<ResultData[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
  const [sortColumn, setSortColumn] = useState<SortColumn | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateReport = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // TODO: Build SQL string
      // Temporary placeholder query
      const sql = `
        SELECT *
        FROM enrollment_predictions
        LIMIT 100;
      `;

      const response = await fetch(`${API_BASE_URL}/api/predict/sql`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          
          sql,
          // model,
          // semesters: selectedSemesters,
          // department: departmentFilter,
          // courseIdentifier: crnFilter,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const json = await response.json();
      console.log("Raw backend response:", json);

      // TODO: Shape backend data into ResultData[]

      const data = json as ResultData[];

      setResults(data);
      setSelectedRows(new Set(data.map((r) => r.id)));
      setShowResults(true);
    } catch (err) {
      console.error("Error generating report:", err);
      setError(
        err instanceof Error ? err.message : "An unknown error occurred.",
      );
      setShowResults(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      // Toggle direction
      setSortDirection(sortDirection === "desc" ? "asc" : "desc");
    } else {
      // New column, default to descending
      setSortColumn(column);
      setSortDirection("desc");
    }
  };

  const getSortedResults = () => {
    if (!sortColumn) return results;

    return [...results].sort((a, b) => {
      let aVal: number | string = a[sortColumn];
      let bVal: number | string = b[sortColumn];

      // For course number, extract numeric part for sorting
      if (sortColumn === "courseNumber") {
        const aNum = parseInt(a.courseNumber.split(" ")[1]);
        const bNum = parseInt(b.courseNumber.split(" ")[1]);
        aVal = aNum;
        bVal = bNum;
      }

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDirection === "desc" ? bVal - aVal : aVal - bVal;
      } else {
        // String comparison
        const aStr = String(aVal).toLowerCase();
        const bStr = String(bVal).toLowerCase();
        if (sortDirection === "desc") {
          return bStr.localeCompare(aStr);
        } else {
          return aStr.localeCompare(bStr);
        }
      }
    });
  };

  const toggleRowSelection = (id: string) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedRows(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedRows.size === results.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(results.map((r) => r.id)));
    }
  };

  const SortIcon = ({ column }: { column: SortColumn }) => {
    if (sortColumn !== column) {
      return <ArrowUpDown className="w-4 h-4 ml-1" />;
    }
    return sortDirection === "desc" ? (
      <ArrowDown className="w-4 h-4 ml-1" />
    ) : (
      <ArrowUp className="w-4 h-4 ml-1" />
    );
  };

  // Placeholder: no real export logic for now
  const handleExport = (format: "csv" | "xlsx") => {
    console.log(`Export (${format}) clicked – logic to be implemented.`);
  };

  const sortedResults = getSortedResults();

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#C2D8FF]/30 to-white">
      <div className="max-w-6xl mx-auto px-6 py-6">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-[#194678] mb-2">Enrollment Predictor</h1>
          <p className="text-[#194678]/70">School of Business</p>
        </div>

        {/* Main Content with Tabs */}
        <div className="bg-white rounded-lg shadow-lg p-8 border-t-4 border-[#194678]">
          <Tabs defaultValue="prediction" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger
                value="prediction"
                className="data-[state=active]:bg-[#194678] data-[state=active]:text-white"
              >
                Prediction Mode
              </TabsTrigger>
              <TabsTrigger
                value="training"
                className="data-[state=active]:bg-[#194678] data-[state=active]:text-white"
              >
                Training Mode
              </TabsTrigger>
            </TabsList>

            <TabsContent value="prediction" className="space-y-6">
              {/* File Uploads */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FileUpload
                  label="Data 1"
                  onFileChange={setFile1}
                  expectedHeaders={[
                    "Placeholder Header 1",
                    "Header 2",
                    "Header 3",
                  ]}
                />
                <FileUpload
                  label="Data 2"
                  onFileChange={setFile2}
                  expectedHeaders={[
                    "Placeholder Header 1",
                    "Header 2",
                    "Header 3",
                  ]}
                />
              </div>

              {/* Dropdowns and Filters */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Model Select */}
                <div className="space-y-2">
                  <label className="text-sm">Model Select</label>
                  <Select value={model} onValueChange={setModel}>
                    <SelectTrigger className="border-gray-300 hover:border-[#94BAEB]">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="linear">Linear</SelectItem>
                      <SelectItem value="random-forest">
                        Random Forest
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Semesters */}
                <SemesterSelector
                  selectedSemesters={selectedSemesters}
                  onSelectionChange={setSelectedSemesters}
                />
              </div>

              {/* Filters Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Department Filter */}
                <div className="space-y-2">
                  <label className="text-sm">Filter by Department</label>
                  <Select
                    value={departmentFilter}
                    onValueChange={setDepartmentFilter}
                  >
                    <SelectTrigger className="border-gray-300 hover:border-[#94BAEB]">
                      <SelectValue placeholder="Select department" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="accounting">Accounting</SelectItem>
                      <SelectItem value="business">Business</SelectItem>
                      <SelectItem value="business-analytics">
                        Business Analytics
                      </SelectItem>
                      <SelectItem value="finance">Finance</SelectItem>
                      <SelectItem value="management-organization">
                        Management &amp; Organization
                      </SelectItem>
                      <SelectItem value="mis">
                        Management Information Systems
                      </SelectItem>
                      <SelectItem value="marketing">Marketing</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Course Identifier Filter */}
                <div className="space-y-2">
                  <label className="text-sm">Filter by Course Identifier</label>
                  <Input
                    type="text"
                    placeholder="Enter course identifier"
                    value={crnFilter}
                    onChange={(e) => setCrnFilter(e.target.value)}
                    className="border-gray-300 hover:border-[#94BAEB] focus:border-[#194678]"
                  />
                </div>
              </div>

              {/* Error message */}
              {error && (
                <div className="text-red-600 text-sm border border-red-300 bg-red-50 px-4 py-2 rounded">
                  Error: {error}
                </div>
              )}

              {/* Generate Button */}
              <div className="flex justify-center">
                <Button
                  onClick={handleGenerateReport}
                  className="bg-[#194678] hover:bg-[#194678]/90 text-white px-8 py-6"
                  disabled={isLoading}
                >
                  {isLoading ? "Generating..." : "Generate Report"}
                </Button>
              </div>

              {/* Results Section */}
              {showResults && (
                <div className="space-y-4">
                  <div className="border border-[#94BAEB] rounded-lg overflow-hidden">
                    <div className="bg-[#194678] text-white px-4 py-3">
                      <h3 className="text-white">Report Results</h3>
                    </div>
                    <div className="p-4 max-h-96 overflow-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="w-12">
                              <Checkbox
                                checked={
                                  selectedRows.size === results.length &&
                                  results.length > 0
                                }
                                onCheckedChange={toggleSelectAll}
                              />
                            </TableHead>
                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("confidenceLevel")}
                            >
                              <div className="flex items-center">
                                Confidence Level %
                                <SortIcon column="confidenceLevel" />
                              </div>
                            </TableHead>
                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("seatsNeeded")}
                            >
                              <div className="flex items-center">
                                Seats Needed
                                <SortIcon column="seatsNeeded" />
                              </div>
                            </TableHead>
                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("courseNumber")}
                            >
                              <div className="flex items-center">
                                Course Number
                                <SortIcon column="courseNumber" />
                              </div>
                            </TableHead>
                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("courseTitle")}
                            >
                              <div className="flex items-center">
                                Course Title
                                <SortIcon column="courseTitle" />
                              </div>
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {sortedResults.map((row) => (
                            <TableRow key={row.id}>
                              <TableCell>
                                <Checkbox
                                  checked={selectedRows.has(row.id)}
                                  onCheckedChange={() =>
                                    toggleRowSelection(row.id)
                                  }
                                />
                              </TableCell>
                              <TableCell>{row.confidenceLevel}%</TableCell>
                              <TableCell>{row.seatsNeeded}</TableCell>
                              <TableCell>{row.courseNumber}</TableCell>
                              <TableCell>{row.courseTitle}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>

                  {/* Export Button (no logic yet) */}
                  <div className="flex justify-end">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="outline"
                          className="border-[#194678] text-[#194678] hover:bg-[#C2D8FF]/20"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          Export ({selectedRows.size} selected)
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={() => handleExport("csv")}
                        >
                          Export as CSV
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => handleExport("xlsx")}
                        >
                          Export as XLSX
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="training" className="py-8">
              <div className="text-center text-gray-500">
                <p>Training Mode - Coming Soon</p>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
