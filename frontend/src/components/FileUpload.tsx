import { useState } from "react";
import { Upload, X, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";

interface FileUploadProps {
  label: string;
  onFileChange: (files: File[]) => void;
  expectedHeaders: string[];
}

type UploadStatus = "idle" | "success" | "error";

export function FileUpload({
  label,
  onFileChange,
  expectedHeaders,
}: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");

  const isValidFile = (file: File) =>
    file.name.toLowerCase().endsWith(".csv") ||
    file.name.toLowerCase().endsWith(".xlsx");

  const autoClearVisualState = () => {
    setTimeout(() => {
      setUploadStatus("idle");
      setErrorMessage(null);
    }, 1000);
  };

  const addFiles = (incoming: FileList | File[]) => {
    const incomingArray = Array.from(incoming);
    const validFiles = incomingArray.filter(isValidFile);
    const invalidFiles = incomingArray.filter((f) => !isValidFile(f));

    // SUCCESS
    if (validFiles.length > 0) {
      setUploadStatus("success");
      setErrorMessage(null);

      autoClearVisualState(); // success fades after x seconds
    }

    // ERROR
    else if (invalidFiles.length > 0) {
      const invalidNames = invalidFiles.map((f) => f.name).join(", ");

      setUploadStatus("error");
      setErrorMessage(
        `Invalid file type: ${invalidNames}. Only .csv or .xlsx files are allowed.`
      );

      autoClearVisualState(); // error clears after x seconds

      return;
    }

    // Merge valid files without duplicates
    const existing = new Set(files.map((f) => f.name));
    const merged = [
      ...files,
      ...validFiles.filter((f) => !existing.has(f.name)),
    ];

    setFiles(merged);
    onFileChange(merged);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) addFiles(e.target.files);
    e.target.value = "";
  };

  const handleClearAll = () => {
    setFiles([]);
    onFileChange([]);
    setUploadStatus("idle");
    setErrorMessage(null);
  };

  // Inline background + border (can't be overridden by index.css)
  const dropzoneStyle: React.CSSProperties = {};

  if (uploadStatus === "success") {
    dropzoneStyle.borderColor = "#22c55e"; // green-500
    dropzoneStyle.backgroundColor = "#f0fdf4"; // green-50
  } else if (uploadStatus === "error") {
    dropzoneStyle.borderColor = "#ef4444"; // red-500
    dropzoneStyle.backgroundColor = "#fef2f2"; // red-50
  } else if (isDragging) {
    dropzoneStyle.backgroundColor = "rgba(194, 216, 255, 0.2)";
  }

  return (
    <div className="space-y-2">
      {/* Label + tooltip */}
      <div className="flex items-center gap-2">
        <label className="text-sm">{label}</label>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                className="text-gray-500 hover:text-gray-700"
              >
                <Info className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>
              <div className="text-sm">
                <p className="mb-1">Expected headers:</p>
                <ul className="list-disc list-inside">
                  {expectedHeaders.map((h, i) => (
                    <li key={i}>{h}</li>
                  ))}
                </ul>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Dropzone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${
            uploadStatus === "idle"
              ? "border-gray-300 hover:border-[#94BAEB]"
              : ""
          }
        `}
        style={dropzoneStyle}
      >
        <input
          type="file"
          multiple
          accept=".csv,.xlsx"
          className="hidden"
          id={`file-input-${label}`}
          onChange={handleFileInput}
        />
        <label htmlFor={`file-input-${label}`} className="cursor-pointer block">
          <Upload className="w-8 h-8 mx-auto mb-2 text-[#194678]" />
          <p className="text-sm text-gray-600">
            Drag & drop or{" "}
            <span className="text-[#194678] underline">browse files</span>
          </p>
          <p className="text-xs text-gray-400 mt-1">
            .csv or .xlsx (multiple allowed)
          </p>
        </label>
      </div>

      {/* Uploaded files */}
      {files.length > 0 && (
        <div className="flex justify-between items-center">
          <p className="text-xs text-gray-600">
            Uploaded: {files.map((f) => f.name).join(", ")}
          </p>
          <button
            onClick={handleClearAll}
            className="text-xs flex items-center gap-1 text-gray-500 hover:text-red-600"
          >
            <X className="w-3 h-3" /> Clear all
          </button>
        </div>
      )}

      {/* Error message (auto-clears after 2s) */}
      {errorMessage && (
        <p className="text-xs text-red-600 mt-1">{errorMessage}</p>
      )}
    </div>
  );
}
