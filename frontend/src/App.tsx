import { useState, useEffect } from "react";
import { FileUpload } from "./components/FileUpload";
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
import { Popover, PopoverContent, PopoverTrigger } from "./components/ui/popover";
import {
  Download,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronDown,
} from "lucide-react";

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

interface SemesterOption {
  term: number;
  term_desc: string;
}

interface DepartmentOption {
  code: string;
  name: string;
}

interface SemesterSelectorProps {
  allSemesters: SemesterOption[];
  selectedSemesters: string[];          // we store the selected term_desc strings
  onSelectionChange: (semesters: string[]) => void;
  isLoading?: boolean;
  error?: string | null;
}

function SemesterSelector({
  allSemesters,
  selectedSemesters,
  onSelectionChange,
  isLoading,
  error,
}: SemesterSelectorProps) {
  const handleToggle = (termDesc: string) => {
    if (selectedSemesters.includes(termDesc)) {
      onSelectionChange(selectedSemesters.filter((s) => s !== termDesc));
    } else {
      onSelectionChange([...selectedSemesters, termDesc]);
    }
  };

  // convenience values
  const totalCount = allSemesters.length;
  const selectedCount = selectedSemesters.length;
  const allSelected = totalCount > 0 && selectedCount === totalCount;
  const partiallySelected = selectedCount > 0 && selectedCount < totalCount;

  const toggleSelectAll = () => {
    if (allSelected) {
      onSelectionChange([]); // clear all
    } else {
      onSelectionChange(allSemesters.map((s) => s.term_desc)); // select all
    }
  };

  const buttonLabel = (() => {
    if (isLoading) return "Loading semesters...";
    if (error) return "Error loading semesters";
    if (selectedSemesters.length > 0) {
      return `${selectedSemesters.length} selected`;
    }
    return "Select semesters";
  })();

  return (
    <div className="space-y-2">
      <label className="text-sm">Select Training Semesters</label>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className="w-full justify-between border-gray-300 hover:border-[#94BAEB]"
            disabled={isLoading || !!error || allSemesters.length === 0}
          >
            <span className="text-sm">{buttonLabel}</span>
            <ChevronDown className="w-4 h-4 opacity-50" />
          </Button>
        </PopoverTrigger>

        <PopoverContent className="w-64 p-4" align="start">
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {/* Select all row */}
            <div className="flex items-center space-x-2 border-b pb-2 mb-2">
              <Checkbox
                id="select-all-semesters"
                checked={allSelected}
                onCheckedChange={toggleSelectAll}
              />
              <label
                htmlFor="select-all-semesters"
                className="text-sm cursor-pointer flex-1"
              >
                {allSelected
                  ? `All selected (${totalCount})`
                  : partiallySelected
                  ? `${selectedCount} of ${totalCount} selected`
                  : "Select all"}
              </label>
            </div>

            {allSemesters.map((semester) => (
              <div key={semester.term} className="flex items-center space-x-2">
                <Checkbox
                  id={semester.term_desc}
                  checked={selectedSemesters.includes(semester.term_desc)}
                  onCheckedChange={() => handleToggle(semester.term_desc)}
                />
                <label
                  htmlFor={semester.term_desc}
                  className="text-sm cursor-pointer flex-1"
                >
                  {semester.term_desc}
                </label>
              </div>
            ))}

            {!isLoading && !error && allSemesters.length === 0 && (
              <div className="text-xs text-gray-500">No semesters found.</div>
            )}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}

// For Docker, set VITE_API_URL in env; for local dev, fallback to localhost
const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL ?? "http://localhost:5000";

export default function App() {
  const [files1, setFiles1] = useState<File[]>([]);
  const [trainingFile, setTrainingFile] = useState<File[]>([]);
  const [model, setModel] = useState<string>("");
  const [selectedSemesters, setSelectedSemesters] = useState<string[]>([]);

  const [semesterOptions, setSemesterOptions] = useState<SemesterOption[]>([]);
  const [isLoadingSemesters, setIsLoadingSemesters] = useState(false);
  const [semestersError, setSemestersError] = useState<string | null>(null);

  const [crnFilter, setCrnFilter] = useState("");
  const [departmentFilter, setDepartmentFilter] = useState<string>("");
  const [departments, setDepartments] = useState<DepartmentOption[]>([]);
  const [isLoadingDepartments, setIsLoadingDepartments] = useState(false);
  const [departmentsError, setDepartmentsError] = useState<string | null>(null);

  type PredictionSeason = "spring" | "fall";

  const [predictionSeason, setPredictionSeason] = useState<PredictionSeason>("spring");
  const [predictionYear, setPredictionYear] = useState<string>("2026"); // default year as string
  // Prediction term (term to predict enrollment for) - default 202640
  const [predictionTerm, setPredictionTerm] = useState<number>(202640);

  const [results, setResults] = useState<ResultData[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [rawResponse, setRawResponse] = useState<any>(null);
  const [rawText, setRawText] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [sortColumn, setSortColumn] = useState<SortColumn | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSemesters = async () => {
      setIsLoadingSemesters(true);
      setSemestersError(null);

      try {
        const res = await fetch(`${API_BASE_URL}/api/semesters`);
        if (!res.ok) {
          throw new Error(`Server responded with status ${res.status}`);
        }

        const data = await res.json();
        if (!data.ok) {
          throw new Error(data.error || "Backend returned ok=false");
        }

        const semesters = (data.semesters ?? []) as SemesterOption[];

        // Sort by term ascending just to be safe (even though SQL already does)
        semesters.sort((a, b) => a.term - b.term);

        setSemesterOptions(semesters);
      } catch (err) {
        console.error("Failed to fetch semesters:", err);
        setSemestersError(
          err instanceof Error ? err.message : "Unknown error fetching semesters",
        );
      } finally {
        setIsLoadingSemesters(false);
      }
    };

    fetchSemesters();
  }, []);

  useEffect(() => {
    const fetchDepartments = async () => {
      setIsLoadingDepartments(true);
      setDepartmentsError(null);

      try {
        const res = await fetch(`${API_BASE_URL}/api/departments`);
        if (!res.ok) {
          throw new Error(`Server responded with status ${res.status}`);
        }

        const data = await res.json();
        if (!data.ok) {
          throw new Error(data.error || "Backend returned ok=false");
        }

        const depts = (data.departments ?? []) as DepartmentOption[];
        setDepartments(depts);
      } catch (err) {
        console.error("Failed to fetch departments:", err);
        setDepartmentsError(
          err instanceof Error ? err.message : "Unknown error fetching departments",
        );
      } finally {
        setIsLoadingDepartments(false);
      }
    };

    fetchDepartments();
  }, []);

  useEffect(() => {
  const year = Number(predictionYear);

  // basic guard so you don't generate NaN terms
  if (!Number.isFinite(year) || predictionYear.trim().length !== 4) return;

  const suffix = predictionSeason === "spring" ? "40" : "10";
  const term = Number(`${year}${suffix}`);

  setPredictionTerm(term);
}, [predictionSeason, predictionYear]);

  const handleGenerateReport = async () => {
    console.log("Selected semesters:", selectedSemesters);
    console.log("Selected department:", departmentFilter || "(none)");
    console.log("Prediction term:", predictionTerm);

    setIsLoading(true);
    setError(null);

    try {
      // Build aggregated SQL query for predictions
      // Group by term, subj, crse and sum enrollment across sections
      let sql = `
        SELECT term, subj, crse, SUM(act) as act, AVG(credits) as credits 
        FROM section_detail_report_sbussection_detail_report_sbus
        WHERE term = ${predictionTerm} GROUP BY term, subj, crse
      `;

      // Apply department filter if selected
      if (departmentFilter) {
        sql += ` AND subj = '${departmentFilter}'`;
      }

      sql += ` GROUP BY term, subj, crse`;
      sql += ` LIMIT 200`; // safety limit

      console.log("Generated SQL:", sql);

      // Build prediction payload
      const payload = {
        sql: sql,
        model_prefix: "enrollment_tree_min_",
        features: "min",
      };

      // Call prediction endpoint
      const response = await fetch(`${API_BASE_URL}/api/predict/sql`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const text = await response.text();
      console.log("Raw response text:", text);
      setRawText(text);

      let json: any;
      try {
        json = JSON.parse(text);
      } catch (err) {
        console.error("Failed to parse JSON from response text", err);
        throw new Error("Backend did not return valid JSON");
      }

      console.log("Parsed prediction response:", json);
      setRawResponse(json);

      if (json.error) {
        throw new Error(json.error);
      }

      // json is an array of prediction results
      const rows = Array.isArray(json) ? json : [];

      const cols = rows.length > 0 ? Object.keys(rows[0]) : [];
      console.log("Columns:", cols);
      console.log("Predictions count:", rows.length);

      setColumns(cols);

      // NOTE: You currently aren't mapping rows -> ResultData anywhere in this file.
      // Leaving results as-is (empty) to match your existing behavior.
      setResults([]);

      setShowResults(true);
    } catch (err) {
      console.error("Error generating predictions:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
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
      let aVal: number | string = (a as any)[sortColumn];
      let bVal: number | string = (b as any)[sortColumn];

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

  const sortedResults = getSortedResults();

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#C2D8FF]/30 to-white">
      <div className="max-w-6xl mx-auto px-6 py-6">
        <div className="w-full mb-6 flex items-center justify-between gap-8">
          {/* Logo */}
          <div className="w-64 h-64 flex items-center justify-start">
            <img className="object-contain" src="CCSU_Logo.svg" alt="SOB Logo" />
          </div>

          {/* Title Block 1 */}
          <div className="w-64 h-64 flex items-center justify-center">
            <div className="text-center">
              <div
                className="text-[#194678]"
                style={{ fontSize: "2rem", fontWeight: 800 }}
              >
                Course Sense
              </div>
              <p className="text-[#194678]/70 text-lg font-medium">
                Enrollment Forecaster
              </p>
            </div>
          </div>

          {/* Title Block 2 */}
          <div className="w-64 h-64 flex items-center justify-center">
            <div className="text-center">{/*Blank for now*/}</div>
          </div>
        </div>

        {/* Main Content with Tabs */}
        <div className="bg-white rounded-lg shadow-lg p-8 border-t-4 border-[#194678]">
          <Tabs defaultValue="prediction" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger
                value="prediction"
                className="data-[state=active]:bg-[#194678] data-[state=active]:text-white"
              >
                Prediction
              </TabsTrigger>
              <TabsTrigger
                value="training"
                className="data-[state=active]:bg-[#194678] data-[state=active]:text-white"
              >
                Training
              </TabsTrigger>
            </TabsList>

            <TabsContent value="prediction" className="space-y-6">
              {/* File Uploads */}
              <div>
                <FileUpload
                  label="Upload Model"
                  onFileChange={setFiles1}
                  description={"Upload .pkl file."}
                  acceptedExtensions={[".pkl"]}
                  limitUpload={true}
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
                      <SelectItem value="random-forest">Random Forest</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Department Filter */}
                <div className="space-y-2">
                  <label className="text-sm">Filter by Department</label>
                  <Select
                    value={departmentFilter}
                    onValueChange={setDepartmentFilter}
                    disabled={
                      isLoadingDepartments ||
                      !!departmentsError ||
                      departments.length === 0
                    }
                  >
                    <SelectTrigger className="border-gray-300 hover:border-[#94BAEB]">
                      <SelectValue
                        placeholder={
                          isLoadingDepartments
                            ? "Loading departments..."
                            : departmentsError
                            ? "Error loading departments"
                            : "Select department"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {departments.map((dept) => (
                        <SelectItem key={dept.code} value={dept.name}>
                          {dept.name}
                        </SelectItem>
                      ))}
                      {!isLoadingDepartments &&
                        !departmentsError &&
                        departments.length === 0 && (
                          <div className="px-2 py-1 text-xs text-gray-500">
                            No departments found.
                          </div>
                        )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Semester Select */}
                <div className="space-y-2">
                  <label className="text-sm">Select Prediction Semester</label>
                  <Select
                    value={predictionSeason}
                    onValueChange={(val) => setPredictionSeason(val as PredictionSeason)}
                  >
                    <SelectTrigger className="border-gray-300 hover:border-[#94BAEB]">
                      <SelectValue placeholder="Select semester" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="spring">Spring</SelectItem>
                      <SelectItem value="fall">Fall</SelectItem>
                    </SelectContent>
                  </Select>
                </div>


                {/* Year Select */}
                <div className="space-y-2">
                  <label className="text-sm">Enter Prediction Year</label>
                  <Input
                    type="text"
                    placeholder="Enter year"
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
                      <div>
                        <h4 className="font-semibold mb-2">Raw Response</h4>
                        {rawText ? (
                          <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto max-h-64">
                            {rawText}
                          </pre>
                        ) : rawResponse ? (
                          <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto max-h-64">
                            {JSON.stringify(rawResponse, null, 2)}
                          </pre>
                        ) : (
                          <div className="text-sm text-gray-500">
                            No response captured yet
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Export Button (all rows by default) */}
                  <div className="flex justify-end">
                    <Button
                      variant="outline"
                      className="border-[#194678] text-[#194678] hover:bg-[#C2D8FF]/20"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Export
                    </Button>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="training" className="py-8">
              <div className="space-y-6">
                {/* Schema Information */}
                <div className="border border-[#94BAEB] rounded-lg overflow-hidden">
                  <div className="bg-[#194678] text-white px-4 py-3">
                    <h3 className="text-white">Schema Information</h3>
                  </div>
                  <div className="p-6 space-y-6">
                    <div className="bg-[#C2D8FF]/20 rounded-md p-4 space-y-3">
                      {/* Feature Description */}
                      <div className="py-2">
                        <p className="text-md mb-1">Feature Description:</p>
                        <p className="text-sm">Text here...</p>
                      </div>
                      {/* Required Fields */}
                      <div className="py-2">
                        <p className="text-md mb-1">Required Fields:</p>
                        <p className="text-sm">
                          Term, Term Desc, CRN, Subj, Crse, Sec, Credits, Title,
                          Act, XL Act
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* File Upload for Training Data */}
                <div>
                  <FileUpload
                    label="Upload Training Data"
                    onFileChange={setTrainingFile}
                    description={
                      "Term, Term Desc, CRN, Subj, Crse, Sec, Credits, Title, Act, XL Act"
                    }
                    acceptedExtensions={[".csv", ".xlsx"]}
                    limitUpload={false}
                  />
                </div>

                {/* Semesters */}
                <SemesterSelector
                  allSemesters={semesterOptions}
                  selectedSemesters={selectedSemesters}
                  onSelectionChange={setSelectedSemesters}
                  isLoading={isLoadingSemesters}
                  error={semestersError}
                />

                {/* Train Model Button */}
                <div className="flex justify-center">
                  <Button
                    className="bg-[#194678] hover:bg-[#194678]/90 text-white px-8 py-6"
                    disabled={!trainingFile}
                  >
                    Train Model
                  </Button>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
