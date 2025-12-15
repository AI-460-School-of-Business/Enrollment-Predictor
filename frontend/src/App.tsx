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

import subjectDepartmentMap from "./subjectDepartmentMap.json";

const SUBJECT_DEPT_MAP = subjectDepartmentMap as Record<string, string>;

interface ResultRow {
  id: string;
  deptName: string;   // mapped readable name
  subj: string;       // actual subj code
  crse: number;
  credits: number;
  prediction: number;
}

type SortColumn =
  | "deptName"
  | "subj"
  | "crse"
  | "credits"
  | "prediction";

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

  const [predictionYear, setPredictionYear] = useState<string>("2026"); // default year as string
  const [predictionSeason, setPredictionSeason] = useState<PredictionSeason>("spring");
  // Prediction term (term to predict enrollment for) - default 202640

  const [results, setResults] = useState<ResultRow[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [rawResponse, setRawResponse] = useState<any>(null);
  const [rawText, setRawText] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [sortColumn, setSortColumn] = useState<SortColumn | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Train Model
  const [isTraining, setIsTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);


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

  const handleGenerateReport = async () => {
    console.log("Selected semesters:", selectedSemesters);
    console.log("Selected department:", departmentFilter || "(none)");
    console.log("Prediction season:", predictionSeason);
    console.log("Prediction year (export label):", predictionYear);


    setIsLoading(true);
    setError(null);

    try {
      // Build aggregated SQL query for predictions
      // Group by term, subj, crse and sum enrollment across sections
      // Build aggregated SQL query for predictions
      const termSuffix = predictionSeason === "spring" ? "40" : "10";

      let sql = `
        SELECT term, subj, crse, SUM(act) AS act, AVG(credits) AS credits
        FROM section_detail_report_sbussection_detail_report_sbus
        WHERE RIGHT(term::text, 2) = '${termSuffix}'
      `;

      // Apply department filter BEFORE GROUP BY
      if (departmentFilter) {
        sql += ` AND subj = '${departmentFilter}'`;
      }

      sql += `
        GROUP BY term, subj, crse
        LIMIT 200
      `;

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

        const rows = Array.isArray(json) ? json : [];

        const mapped: ResultRow[] = rows.map((r: any) => {
          const subjCode = String(r.subj ?? "").trim().toUpperCase();
          const deptName = SUBJECT_DEPT_MAP[subjCode] ?? subjCode;

          return {
            id: `${r.term}-${subjCode}-${r.crse}`,  // stable unique id
            deptName,
            subj: subjCode,
            crse: Number(r.crse),
            credits: Number(r.credits),
            prediction: Number(r.prediction),
          };
        });

        setResults(mapped);
        setShowResults(true);

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
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];

      // numeric sort
      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDirection === "desc" ? bVal - aVal : aVal - bVal;
      }

      // string sort
      const aStr = String(aVal).toLowerCase();
      const bStr = String(bVal).toLowerCase();

      return sortDirection === "desc"
        ? bStr.localeCompare(aStr)
        : aStr.localeCompare(bStr);
    });
  };

  const SortIcon = ({ column }: { column: SortColumn }) => {
    if (sortColumn !== column) {
      return <ArrowUpDown className="w-4 h-4 ml-1 opacity-50" />;
    }
    return sortDirection === "desc" ? (
      <ArrowDown className="w-4 h-4 ml-1" />
    ) : (
      <ArrowUp className="w-4 h-4 ml-1" />
    );
  };

  const sortedResults = getSortedResults();

      const handleTrainModel = async () => {
    // Framework only, no training logic yet
    console.log("Train Model clicked");
    console.log("Training files:", trainingFile);
    console.log("Selected semesters:", selectedSemesters);

    setIsTraining(true);
    setTrainingError(null);

    try {
      // TODO: add training logic here 

    } catch (err) {
      console.error("Error training model:", err);
      setTrainingError(err instanceof Error ? err.message : "Unknown training error");
    } finally {
      setIsTraining(false);
    }
  };

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
                        <SelectItem key={dept.code} value={dept.code}>
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
                    inputMode="numeric"
                    placeholder="Enter year for export title (e.g., 2026)"
                    value={predictionYear}
                    onChange={(e) => {
                      const digitsOnly = e.target.value.replace(/\D/g, "").slice(0, 4);
                      setPredictionYear(digitsOnly);
                    }}
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
                      <h3 className="text-white">Prediction Results</h3>
                    </div>

                    <div className="p-4 max-h-96 overflow-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("deptName")}
                            >
                              <div className="flex items-center">
                                Department
                                <SortIcon column="deptName" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("subj")}
                            >
                              <div className="flex items-center">
                                Subj
                                <SortIcon column="subj" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("crse")}
                            >
                              <div className="flex items-center">
                                Crse
                                <SortIcon column="crse" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("credits")}
                            >
                              <div className="flex items-center">
                                Credits
                                <SortIcon column="credits" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("prediction")}
                            >
                              <div className="flex items-center">
                                Prediction
                                <SortIcon column="prediction" />
                              </div>
                            </TableHead>
                          </TableRow>
                        </TableHeader>

                        <TableBody>
                          {sortedResults.map((row) => (
                            <TableRow key={row.id}>
                              <TableCell>{row.deptName}</TableCell>
                              <TableCell className="font-mono">{row.subj}</TableCell>
                              <TableCell className="font-mono">{row.crse}</TableCell>
                              <TableCell>{Number.isFinite(row.credits) ? row.credits.toFixed(2) : row.credits}</TableCell>
                              <TableCell>{Number.isFinite(row.prediction) ? row.prediction.toFixed(2) : row.prediction}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
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
                    onClick={handleTrainModel}
                    className="bg-[#194678] hover:bg-[#194678]/90 text-white px-8 py-6"
                    disabled={isTraining || trainingFile.length === 0}
                  >
                    {isTraining ? "Training..." : "Train Model"}
                  </Button>
                </div>

                {/* Optional error display (framework) */}
                {trainingError && (
                  <div className="text-red-600 text-sm border border-red-300 bg-red-50 px-4 py-2 rounded">
                    Error: {trainingError}
                  </div>
                )}

              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
