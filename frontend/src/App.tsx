/**
 * App.tsx
 * -----------------------------------------------------------------------------
 * Course Sense: Enrollment Forecaster UI
 *
 * Responsibilities:
 * - Fetch reference data (semesters, departments) from the backend.
 * - Allow users to generate enrollment predictions via SQL-based prediction endpoint.
 * - Display sortable results and provide an export button hook.
 * - Provide a training tab with file upload + semester selection + "Train Model" hook.
 *
 * Notes:
 * - This file intentionally contains "framework-only" handlers for export + training.
 * - Consider extracting large UI sections into smaller components as this grows.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { FileUpload } from "./components/FileUpload";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { SelectField } from "./components/SelectField";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Checkbox } from "./components/ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "./components/ui/popover";

import { Download, ArrowUpDown, ArrowUp, ArrowDown, ChevronDown } from "lucide-react";

import subjectDepartmentMap from "./subjectDepartmentMap.json";

/** Subject → Department name mapping (e.g., "FIN" → "Finance"). */
const SUBJECT_DEPT_MAP = subjectDepartmentMap as Record<string, string>;

/**
 * Base URL for backend API.
 * - For Docker, set VITE_API_URL in env
 * - For local dev, fallback to localhost
 */
const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL ?? "http://localhost:5000";

declare global {
  interface Window {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
  }
}

/** UI row for the prediction table. */
interface ResultRow {
  /** Stable unique id for React keys. */
  id: string;
  /** Human-readable department name from SUBJECT_DEPT_MAP. */
  deptName: string;
  /** Subject code (e.g., FIN). */
  subj: string;
  /** Course number (e.g., 330). */
  crse: number;
  /** Average credits across sections in the aggregate query. */
  credits: number;
  /** Model prediction result (expected enrollment). */
  prediction: number;
}

type SortColumn = "deptName" | "subj" | "crse" | "credits" | "prediction";
type SortDirection = "asc" | "desc";

interface SemesterOption {
  term: number;
  term_desc: string;
}

interface DepartmentOption {
  code: string;
  name: string;
}

type PredictionSeason = "spring" | "fall";

/**
 * Multi-select semester picker used in the Training tab.
 * Stores `term_desc` strings as the selection value (matches your current usage).
 */
interface SemesterSelectorProps {
  allSemesters: SemesterOption[];
  selectedSemesters: string[]; // stored as term_desc strings
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
  /** Toggle a single semester term_desc in/out of the selection. */
  const handleToggle = (termDesc: string) => {
    if (selectedSemesters.includes(termDesc)) {
      onSelectionChange(selectedSemesters.filter((s) => s !== termDesc));
      return;
    }
    onSelectionChange([...selectedSemesters, termDesc]);
  };

  // Convenience values for the "Select all" row
  const totalCount = allSemesters.length;
  const selectedCount = selectedSemesters.length;
  const allSelected = totalCount > 0 && selectedCount === totalCount;
  const partiallySelected = selectedCount > 0 && selectedCount < totalCount;

  /** Toggle all semesters on/off. */
  const toggleSelectAll = () => {
    if (allSelected) {
      onSelectionChange([]); // clear all
      return;
    }
    onSelectionChange(allSemesters.map((s) => s.term_desc)); // select all
  };

  /** Button label state for the Popover trigger. */
  const buttonLabel = (() => {
    if (isLoading) return "Loading semesters...";
    if (error) return "Error loading semesters";
    if (selectedSemesters.length > 0) return `${selectedSemesters.length} selected`;
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

/**
 * App Component
 * -----------------------------------------------------------------------------
 * Provides two main tabs:
 * - Prediction: uploads a model file + filters + generates predictions (SQL-based)
 * - Training: uploads training data + chooses semesters + trains a model (framework)
 */
export default function App() {
  // -----------------------------
  // Upload State
  // -----------------------------
  const [modelFiles, setModelFiles] = useState<File[]>([]);
  const [trainingFiles, setTrainingFiles] = useState<File[]>([]);

  // -----------------------------
  // Prediction Form State
  // -----------------------------
  const [model, setModel] = useState<string>("");
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [isUploadingModel, setIsUploadingModel] = useState(false);
  const [uploadedModelFilename, setUploadedModelFilename] = useState<string | null>(null);
  const [departmentFilter, setDepartmentFilter] = useState<string>("");
  const [predictionSeason, setPredictionSeason] = useState<PredictionSeason>("spring");
  const [predictionYear, setPredictionYear] = useState<string>("2026");
  const [accuracyPath, setAccuracyPath] = useState<string>("");
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  // -----------------------------
  // Training Form State
  // -----------------------------
  const [selectedSemesters, setSelectedSemesters] = useState<string[]>([]);
  const [trainingModel, setTrainingModel] = useState<string>("tree");
  const [trainingFeatures, setTrainingFeatures] = useState<string>("min");
  const [isUploadingTraining, setIsUploadingTraining] = useState(false);
  const [trainingUploadError, setTrainingUploadError] = useState<string | null>(null);

  // -----------------------------
  // Reference Data State
  // -----------------------------
  const [semesterOptions, setSemesterOptions] = useState<SemesterOption[]>([]);
  const [isLoadingSemesters, setIsLoadingSemesters] = useState(false);
  const [semestersError, setSemestersError] = useState<string | null>(null);

  const [departments, setDepartments] = useState<DepartmentOption[]>([]);
  const [isLoadingDepartments, setIsLoadingDepartments] = useState(false);
  const [departmentsError, setDepartmentsError] = useState<string | null>(null);

  // -----------------------------
  // Results State
  // -----------------------------
  const [results, setResults] = useState<ResultRow[]>([]);
  const [showResults, setShowResults] = useState(false);

  // Sorting
  const [sortColumn, setSortColumn] = useState<SortColumn | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  // Prediction request state
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Training request state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  // Debug / diagnostics (optional)
  const [rawResponse, setRawResponse] = useState<any>(null);
  const [rawText, setRawText] = useState<string | null>(null);

  // -----------------------------
  // Effects: Fetch reference data
  // -----------------------------
  useEffect(() => {
    const fetchSemesters = async () => {
      setIsLoadingSemesters(true);
      setSemestersError(null);

      try {
        const res = await fetch(`${API_BASE_URL}/api/semesters`);
        if (!res.ok) throw new Error(`Server responded with status ${res.status}`);

        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "Backend returned ok=false");

        const semesters = (data.semesters ?? []) as SemesterOption[];

        // Sort by term ascending (defensive; SQL likely already sorts)
        semesters.sort((a, b) => a.term - b.term);
        setSemesterOptions(semesters);
      } catch (err) {
        console.error("Failed to fetch semesters:", err);
        setSemestersError(err instanceof Error ? err.message : "Unknown error fetching semesters");
      } finally {
        setIsLoadingSemesters(false);
      }
    };

    fetchSemesters();
  }, []);

  const fetchModels = useCallback(async () => {
    setIsLoadingModels(true);
    setModelError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/models`);
      if (!res.ok) throw new Error(`Server responded with status ${res.status}`);

      const data = await res.json();
      const filenames = Array.isArray(data.models)
        ? data.models
            .map((m: any) => (typeof m === "string" ? m : m?.filename))
            .filter(Boolean)
        : [];

      setModelOptions(filenames);
    } catch (err) {
      console.error("Failed to fetch models:", err);
      setModelError(err instanceof Error ? err.message : "Unknown error fetching models");
    } finally {
      setIsLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  useEffect(() => {
    const fetchDepartments = async () => {
      setIsLoadingDepartments(true);
      setDepartmentsError(null);

      try {
        const res = await fetch(`${API_BASE_URL}/api/departments`);
        if (!res.ok) throw new Error(`Server responded with status ${res.status}`);

        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "Backend returned ok=false");

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

  // ----------------------------- 
  // Helpers: Sorting
  // -----------------------------
  /**
   * Handle sort state transitions:
   * - If clicking same column: toggle direction.
   * - If clicking new column: set it and default direction to "desc".
   */
  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      setSortDirection((prev) => (prev === "desc" ? "asc" : "desc"));
      return;
    }
    setSortColumn(column);
    setSortDirection("desc");
  };

  /**
   * Compute sorted rows.
   * useMemo keeps render work low when results are large.
   */
  const sortedResults = useMemo(() => {
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
  }, [results, sortColumn, sortDirection]);

  /**
   * Sort icon helper for table headers.
   * Defined inline for readability; could be extracted to its own component.
   */
  const SortIcon = ({ column }: { column: SortColumn }) => {
    if (sortColumn !== column) return <ArrowUpDown className="w-4 h-4 ml-1 opacity-50" />;
    return sortDirection === "desc" ? (
      <ArrowDown className="w-4 h-4 ml-1" />
    ) : (
      <ArrowUp className="w-4 h-4 ml-1" />
    );
  };

  const handleModelUploadChange = async (files: File[]) => {
    setModelFiles(files);
    if (!files.length) return;

    const file = files[0];
    const formData = new FormData();
    formData.append("model", file);

    setIsUploadingModel(true);
    setModelError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/models`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(data?.error || `Server responded with status ${res.status}`);
      }

      const savedFilename = data?.filename;

      await fetchModels();
      if (savedFilename) {
        setUploadedModelFilename(savedFilename);
      }
    } catch (err) {
      console.error("Failed to upload model:", err);
      setModelError(err instanceof Error ? err.message : "Unknown error uploading model");
    } finally {
      setIsUploadingModel(false);
    }
  };

  const handleTrainingUploadChange = async (files: File[]) => {
    setTrainingFiles(files);
    setTrainingUploadError(null);
    if (!files.length) return;

    setIsUploadingTraining(true);
    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${API_BASE_URL}/api/train/upload-file`, {
          method: "POST",
          body: formData,
        });

        const data = await res.json().catch(() => null);
        if (!res.ok || data?.ok === false) {
          throw new Error(data?.error || `Failed to upload ${file.name}`);
        }
      }
    } catch (err) {
      console.error("Failed to upload training file:", err);
      setTrainingUploadError(err instanceof Error ? err.message : "Unknown upload error");
    } finally {
      setIsUploadingTraining(false);
    }
  };

  // -----------------------------
  // Actions: Prediction
  // -----------------------------
  /**
   * Generate prediction report:
   * - Builds an aggregate SQL query (term/subj/crse)
   * - Sends it to /api/predict/sql
   * - Maps the response into ResultRow[]
   */
  const handleGenerateReport = async () => {
    console.log("Selected department:", departmentFilter || "(none)");
    console.log("Prediction season:", predictionSeason);
    console.log("Prediction year:", predictionYear);

    setIsGenerating(true);
    setError(null);

  try {
    /**
     * 
     * term / 100 extracts the year portion (YYYY)
     * This matches the training query exactly
     * 
     */
    const termSuffix = predictionSeason === "spring" ? "40" : "10";

    let sql = `
      SELECT term, subj, crse, SUM(act) AS act, AVG(credits) AS credits
      FROM section_detail_report_sbussection_detail_report_sbus
      WHERE (term / 100) = 2025 AND RIGHT(term::text, 2) = '${termSuffix}'
    `;

    // Optional department filter
    if (departmentFilter) {
      sql += ` AND subj = '${departmentFilter}'`;
    }

    sql += `
      GROUP BY term, subj, crse
      HAVING SUM(act) >= 10
    `;

    console.log("Generated SQL:", sql);

    const payload = {
      sql,
      model_prefix: "enrollment_tree_min_",
      features: "min",
    } as Record<string, string>;

    const modelFilename = uploadedModelFilename || model;
    if (modelFilename) {
      payload.model_filename = modelFilename;
    }


      const response = await fetch(`${API_BASE_URL}/api/predict/sql`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error(`Server responded with status ${response.status}`);

      const text = await response.text();
      setRawText(text);

      let json: any;
      try {
        json = JSON.parse(text);
      } catch (parseErr) {
        console.error("Failed to parse JSON from response text", parseErr);
        throw new Error("Backend did not return valid JSON");
      }

      setRawResponse(json);

      // If backend uses { error: "..."} shape sometimes
      if (json?.error) throw new Error(json.error);

      const rows = Array.isArray(json) ? json : [];

      const mapped: ResultRow[] = rows.map((r: any) => {
        const subjCode = String(r.subj ?? "").trim().toUpperCase();
        const deptName = SUBJECT_DEPT_MAP[subjCode] ?? subjCode;

        return {
          id: `${r.term}-${subjCode}-${r.crse}`, // stable unique id
          deptName,
          subj: subjCode,
          crse: Number(r.crse),
          credits: Number(r.credits),
          prediction: Number(r.prediction),
        };
      });

      setResults(mapped);
      setShowResults(true);
    } catch (err) {
      console.error("Error generating predictions:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
      setShowResults(false);
      setResults([]);
    } finally {
      setIsGenerating(false);
    }
  };

  // -----------------------------
  // Actions: Export (framework)
  // -----------------------------
  /**
   * Export handler (framework-only).
   * Implement CSV/XLSX export here, likely based on `results` or `sortedResults`.
   */
  const handleExportResults = async () => {
    if (!rawResponse || !Array.isArray(rawResponse) || rawResponse.length === 0) {
      setExportError("No data available to export.");
      return;
    }

    setIsExporting(true);
    setExportError(null);

    try {
      const modelInfo = {
        model_type: "Tree Ensemble",
        model_name: uploadedModelFilename || model || "enrollment_tree_v2",
        feature_schema: "auto",
      };

      const res = await fetch(`${API_BASE_URL}/api/reports/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rows: rawResponse,
          accuracy_csv: accuracyPath || null,
          model_info: modelInfo,
        }),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(text || `Server responded with status ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const disposition = res.headers.get("Content-Disposition") || "";
      const match = disposition.match(/filename="?([^\";]+)"?/i);
      const filename = match?.[1] ?? "enrollment_report.xlsx";

      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();

      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
      setExportError(err instanceof Error ? err.message : "Unknown export error");
    } finally {
      setIsExporting(false);
    }
  };

  const handleAccuracyPathInputChange = (value: string) => {
    setAccuracyPath(value);
    if (browseError) {
      setBrowseError(null);
    }
  };

  const handleAccuracyPathBrowse = async () => {
    setBrowseError(null);

    try {
      if (typeof window === "undefined" || !window.showDirectoryPicker) {
        setBrowseError("Folder browsing is not available in this browser.");
        return;
      }

      const directoryHandle = await window.showDirectoryPicker();
      if (!directoryHandle || !directoryHandle.name) return;

      handleAccuracyPathInputChange(directoryHandle.name);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      console.error("Directory picker failed:", err);
      setBrowseError("Unable to open the folder picker. Please try again.");
    }
  };

  // -----------------------------
  // Actions: Training (framework)
  // -----------------------------
  /**
   * Train model handler (framework-only).
   * Suggested future implementation:
   * - Build FormData with training files + semester selection
   * - POST to `/api/train` (or similar)
   * - Show status + store returned model artifact reference
   */
  const handleTrainModel = async () => {
    console.log("Train Model clicked");
    console.log("Training files:", trainingFiles);
    console.log("Selected semesters:", selectedSemesters);

    setIsTraining(true);
    setTrainingError(null);

    try {
      const selectedTerms = semesterOptions
        .filter((s) => selectedSemesters.includes(s.term_desc))
        .map((s) => s.term);

      if (selectedTerms.length === 0) {
        throw new Error("Select at least one semester.");
      }

      const payload = {
        model: trainingModel,
        features: trainingFeatures,
        terms: selectedTerms,
      };

      const res = await fetch(`${API_BASE_URL}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json().catch(() => null);
      if (!res.ok || data?.ok === false) {
        throw new Error(data?.error || `Server responded with status ${res.status}`);
      }

      console.log("Training complete:", data);

      // Refresh available models and preselect the newly trained one
      await fetchModels();
      if (data?.model_filename) {
        setUploadedModelFilename(data.model_filename);
      }
    } catch (err) {
      console.error("Error training model:", err);
      setTrainingError(err instanceof Error ? err.message : "Unknown training error");
    } finally {
      setIsTraining(false);
    }
  };

  // -----------------------------
  // Render
  // -----------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-[#C2D8FF]/30 to-white">
      <div className="max-w-6xl mx-auto px-6 py-6">
        {/* Header */}
        <div className="w-full mb-6 flex items-center justify-between gap-8">
          {/* Logo */}
          <div className="w-64 h-64 flex items-center justify-start">
            <img className="object-contain" src="CCSU_Logo.svg" alt="SOB Logo" />
          </div>

          {/* Title Block */}
          <div className="w-64 h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-[#194678]" style={{ fontSize: "2rem", fontWeight: 800 }}>
                Course Sense
              </div>
              <p className="text-[#194678]/70 text-lg font-medium">
                Enrollment Forecaster
              </p>
            </div>
          </div>

          {/* Placeholder (for symmetry / future content) */}
          <div className="w-64 h-64 flex items-center justify-center">
            <div className="text-center">{/* Blank for now */}</div>
          </div>
        </div>

        {/* Main Content */}
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

              {/* ------------------------ Prediction Tab ------------------------ */}
              <TabsContent value="prediction" className="space-y-6">
                {/* Upload Model */}
                <FileUpload
                  label="Upload Model"
                  onFileChange={handleModelUploadChange}
                  description="Upload a .pkl model to make it available in the dropdown below."
                  acceptedExtensions={[".pkl"]}
                  limitUpload={true}
                />

                {/* Dropdowns */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Model Select (available .pkl files in container) */}
                  <div className="space-y-2">
                    <label className="text-sm">Model Select</label>
                    <Select
                      value={model}
                      onValueChange={setModel}
                      disabled={isLoadingModels || isUploadingModel || modelOptions.length === 0}
                    >
                      <SelectTrigger className="border-gray-300 hover:border-[#94BAEB]">
                        <SelectValue
                          placeholder={
                            isLoadingModels
                              ? "Loading models..."
                              : modelError
                              ? "Error loading models"
                              : "Select Model"
                          }
                        />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map((filename) => (
                          <SelectItem key={filename} value={filename}>
                            {filename}
                          </SelectItem>
                        ))}

                        {!isLoadingModels && modelOptions.length === 0 && (
                          <div className="px-2 py-1 text-xs text-gray-500">No models found.</div>
                        )}
                      </SelectContent>
                    </Select>

                    {(isUploadingModel || isLoadingModels) && (
                      <p className="text-xs text-gray-500">
                        {isUploadingModel ? "Uploading model..." : "Loading models..."}
                      </p>
                    )}

                    {modelError && (
                      <p className="text-xs text-red-600">Model error: {modelError}</p>
                    )}
                  </div>

                {/* Department Filter */}
                <div className="space-y-2">
                  <label className="text-sm">Filter by Department (Optional)</label>
                  <Select
                    value={departmentFilter}
                    onValueChange={setDepartmentFilter}
                    disabled={
                      isLoadingDepartments || !!departmentsError || departments.length === 0
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

                      {!isLoadingDepartments && !departmentsError && departments.length === 0 && (
                        <div className="px-2 py-1 text-xs text-gray-500">
                          No departments found.
                        </div>
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Term Inputs */}
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

                {/* Year Input */}
                <div className="space-y-2">
                  <label className="text-sm">Enter Prediction Year</label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    placeholder="Enter prediction year"
                    value={predictionYear}
                    onChange={(e) => {
                      // Keep only up to 4 digits
                      const digitsOnly = e.target.value.replace(/\D/g, "").slice(0, 4);
                      setPredictionYear(digitsOnly);
                    }}
                    className="border-gray-300 hover:border-[#94BAEB] focus:border-[#194678]"
                  />
                </div>
              </div>

              {/* Error Message */}
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
                  disabled={isGenerating}
                >
                  {isGenerating ? "Generating..." : "Generate Report"}
                </Button>
              </div>

              {/* Results */}
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
                                Department <SortIcon column="deptName" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("subj")}
                            >
                              <div className="flex items-center">
                                Subj <SortIcon column="subj" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("crse")}
                            >
                              <div className="flex items-center">
                                Crse <SortIcon column="crse" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("credits")}
                            >
                              <div className="flex items-center">
                                Credits <SortIcon column="credits" />
                              </div>
                            </TableHead>

                            <TableHead
                              className="cursor-pointer hover:bg-gray-50"
                              onClick={() => handleSort("prediction")}
                            >
                              <div className="flex items-center">
                                Prediction <SortIcon column="prediction" />
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
                              <TableCell>
                                {Number.isFinite(row.credits)
                                  ? row.credits.toFixed(2)
                                  : row.credits}
                              </TableCell>
                              <TableCell>
                                {Number.isFinite(row.prediction)
                                  ? row.prediction.toFixed(2)
                                  : row.prediction}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>

                  {/* Export */}
                  <div className="flex flex-col md:flex-row md:items-end gap-4">
                    <div className="flex-1 md:flex-none space-y-2">
                      <label className="text-sm">Accuracy CSV Path (optional)</label>
                      <div className="flex gap-2">
                        <Input
                          type="text"
                          placeholder="backend/app/ml/test_results/..."
                          value={accuracyPath}
                          onChange={(e) => handleAccuracyPathInputChange(e.target.value)}
                          className="flex-1 border-gray-300 hover:border-[#94BAEB] focus:border-[#194678]"
                        />
                        <Button
                          type="button"
                          variant="outline"
                          onClick={handleAccuracyPathBrowse}
                          className="border-[#194678] text-[#194678] hover:bg-[#C2D8FF]/20 whitespace-nowrap"
                        >
                          Browse
                        </Button>
                      </div>
                      {accuracyPath.trim() !== "" && (
                        <p className="text-xs text-[#194678] break-all">
                          Selected path: {accuracyPath}
                        </p>
                      )}
                      {browseError && (
                        <p className="text-xs text-red-600">{browseError}</p>
                      )}
                      <div className="flex justify-end">
                        <Button
                          onClick={handleExportResults}
                          variant="outline"
                          className="border-[#194678] text-[#194678] hover:bg-[#C2D8FF]/20 px-4 py-2 text-sm"
                          disabled={results.length === 0 || isExporting}
                        >
                          <Download className="w-4 h-4 mr-2" />
                          {isExporting ? "Exporting..." : "Export"}
                        </Button>
                      </div>
                      {exportError && (
                        <p className="text-xs text-red-600 text-right">
                          Export error: {exportError}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            {/* ------------------------ Training Tab ------------------------ */}
            <TabsContent value="training" className="py-8">
              <div className="space-y-6">
                {/* Schema Information */}
                <div className="border border-[#94BAEB] rounded-lg overflow-hidden">
                  <div className="bg-[#194678] text-white px-4 py-3">
                    <h3 className="text-white">Training Instructions</h3>
                  </div>

                  <div className="p-6 space-y-6">
                    <div className="bg-[#C2D8FF]/20 rounded-md p-4 space-y-3">
                      <div className="py-2">
                        <p className="text-md mb-1">
                          Load a CSV with the above schema to train a new enrollment prediction model with the given parameters.
                        </p>
                      </div>

                      <div className="py-2">
                        <p className="text-md mb-1"><strong>Enrollment Headcount Schema:</strong></p>
                        <p className="text-md mb-1">Term, Term Desc, CRN, Subj, Crse, Sec, Credits, Title, Act, XL Act </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Training File Upload */}
                <FileUpload
                  label="Upload Training Data"
                  onFileChange={handleTrainingUploadChange}
                  description="Upload data from registrar. Expected headers: Term, Term Desc, CRN, Subj, Crse, Sec, Credits, Title, Act, XL Act."
                  acceptedExtensions={[".csv"]}
                  limitUpload={false}
                />
                {isUploadingTraining && (
                  <p className="text-xs text-gray-500">Uploading training file(s)...</p>
                )}
                {trainingUploadError && (
                  <p className="text-xs text-red-600">Upload error: {trainingUploadError}</p>
                )}

                {/* Training Parameters */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <SelectField
                      label="Model Type"
                      description="Choose which model family to train (Tree is usually fastest to start with)."
                    >
                      <Select value={trainingModel} onValueChange={setTrainingModel}>
                        <SelectTrigger className="border-gray-300 hover:border-[#94BAEB]">
                          <SelectValue placeholder="Select model type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="tree">Tree</SelectItem>
                          <SelectItem value="linear">Linear</SelectItem>
                          <SelectItem value="neural">Neural</SelectItem>
                        </SelectContent>
                      </Select>
                    </SelectField>
                  </div>

                  <div className="space-y-2">
                    <SelectField
                      label="Feature Schema"
                      description="Choose which feature schema to use for training."
                    >
                    <Select value={trainingFeatures} onValueChange={setTrainingFeatures}>
                      <SelectTrigger className="border-gray-300 hover:border-[#94BAEB]">
                        <SelectValue placeholder="Select features" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="min">Min</SelectItem>
                        <SelectItem value="rich">Rich</SelectItem>
                      </SelectContent>
                    </Select>
                    </SelectField>
                  </div>
                </div>

                {/* Semester Multi-Select */}
                <SemesterSelector
                  allSemesters={semesterOptions}
                  selectedSemesters={selectedSemesters}
                  onSelectionChange={setSelectedSemesters}
                  isLoading={isLoadingSemesters}
                  error={semestersError}
                />

                {/* Train Button */}
                <div className="flex justify-center">
                  <Button
                    onClick={handleTrainModel}
                    className="bg-[#194678] hover:bg-[#194678]/90 text-white px-8 py-6"
                    disabled={isTraining || isUploadingTraining || trainingFiles.length === 0}
                  >
                    {isTraining ? "Training..." : "Train Model"}
                  </Button>
                </div>

                {/* Training error (framework) */}
                {trainingError && (
                  <div className="text-red-600 text-sm border border-red-300 bg-red-50 px-4 py-2 rounded">
                    Error: {trainingError}
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>

          {/* Debug panel (optional): uncomment if needed */}
          {/* <pre className="mt-6 text-xs bg-gray-50 p-3 rounded border overflow-auto">
            rawText: {rawText}
            {"\n\n"}
            rawResponse: {JSON.stringify(rawResponse, null, 2)}
          </pre> */}
        </div>
      </div>
    </div>
  );
}
