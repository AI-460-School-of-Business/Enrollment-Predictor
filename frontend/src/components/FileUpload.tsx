import { useState } from 'react';
import { Upload, X, Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';

interface FileUploadProps {
  label: string;
  onFileChange: (file: File | null) => void;
  expectedHeaders: string[];
}

export function FileUpload({ label, onFileChange, expectedHeaders }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.name.endsWith('.csv') || droppedFile.name.endsWith('.xlsx'))) {
      setFile(droppedFile);
      onFileChange(droppedFile);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      onFileChange(selectedFile);
    }
  };

  const handleRemove = () => {
    setFile(null);
    onFileChange(null);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="text-sm">{label}</label>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button type="button" className="text-gray-500 hover:text-gray-700">
                <Info className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>
              <div className="text-sm">
                <p className="mb-1">Expected headers:</p>
                <ul className="list-disc list-inside">
                  {expectedHeaders.map((header, index) => (
                    <li key={index}>{header}</li>
                  ))}
                </ul>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {!file ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
            isDragging
              ? 'border-[#194678] bg-[#C2D8FF]/20'
              : 'border-gray-300 hover:border-[#94BAEB]'
          }`}
        >
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={handleFileInput}
            className="hidden"
            id={`file-${label}`}
          />
          <label htmlFor={`file-${label}`} className="cursor-pointer">
            <Upload className="w-8 h-8 mx-auto mb-2 text-[#194678]" />
            <p className="text-sm text-gray-600">
              Drag & drop or <span className="text-[#194678] underline">browse</span>
            </p>
            <p className="text-xs text-gray-400 mt-1">.csv or .xlsx files</p>
          </label>
        </div>
      ) : (
        <div className="border border-gray-300 rounded-lg p-4 flex items-center justify-between bg-[#C2D8FF]/10">
          <div className="flex items-center gap-2">
            <Upload className="w-4 h-4 text-[#194678]" />
            <span className="text-sm">{file.name}</span>
          </div>
          <button
            onClick={handleRemove}
            className="text-gray-500 hover:text-red-600 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
