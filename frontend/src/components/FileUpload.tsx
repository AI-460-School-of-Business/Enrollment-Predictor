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
  description?: string;
  /** Allowed file extensions, e.g. [".csv", ".xlsx"] */
  acceptedExtensions: string[];
  /** If true, only one file may be uploaded at a time */
  limitUpload?: boolean;
}

type UploadStatus = "idle" | "success" | "error";

export function FileUpload({
  label,
  onFileChange,
  description,
  acceptedExtensions,
  limitUpload = false,
}: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");

  // Normalize extensions: ensure they start with "." and are lowercase
  const normalizedExtensions = acceptedExtensions.map((ext) => {
    const trimmed = ext.trim().toLowerCase();
    return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
  });

  const isValidFile = (file: File) =>
    normalizedExtensions.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );

  const autoClearVisualState = () => {
    // Clear the visual state (green/red border/fill), keep any error text
    setTimeout(() => {
      setUploadStatus("idle");
      setIsDragging(false);
    }, 1000);
  };

  const addFiles = (incoming: FileList | File[]) => {
    const incomingArray = Array.from(incoming);

    // Reject multi-file drop when limitUpload === true ---
    if (limitUpload && incomingArray.length > 1) {
      setUploadStatus("error");
      setErrorMessage("Only one file may be uploaded at a time.");
      autoClearVisualState();
      return;
    }

    const validFiles = incomingArray.filter(isValidFile);
    const invalidFiles = incomingArray.filter((f) => !isValidFile(f));

    // SUCCESS
    if (validFiles.length > 0) {
      setUploadStatus("success");
      setErrorMessage(null);
      autoClearVisualState();
    }

    // ERROR: invalid files only
    if (validFiles.length === 0 && invalidFiles.length > 0) {
      const invalidNames = invalidFiles.map((f) => f.name).join(", ");
      setUploadStatus("error");
      setErrorMessage(
        `Invalid file type: ${invalidNames}. Only ${normalizedExtensions.join(
          ", "
        )} files are allowed.`
      );
      autoClearVisualState();
      return;
    }

    // LIMIT UPLOAD MODE — allow only one file & replace previous
    if (limitUpload) {
      const single = validFiles[0];
      if (!single) return;
      setFiles([single]);
      onFileChange([single]);
      return;
    }

    // MULTI-FILE MODE — merge, avoid duplicates
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

  // Build accept string for <input>, e.g. ".csv,.xlsx"
  const acceptAttr = normalizedExtensions.join(",");

  // Inline background + border
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
      {/* Label + optional tooltip */}
      <div className="flex items-center gap-2">
        <label className="text-sm">{label}</label>
        {description && (
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
                <div className="text-sm max-w-xs">{description}</div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
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
          multiple={!limitUpload}
          accept={acceptAttr}
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
            Accepted: {normalizedExtensions.join(", ")}{" "}
            {limitUpload ? "(single file)" : "(multiple allowed)"}
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

      {/* Error message (stays until valid upload) */}
      {errorMessage && (
        <p className="text-xs text-red-600 mt-1">{errorMessage}</p>
      )}
    </div>
  );
}
