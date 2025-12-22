import * as React from "react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";

type SelectFieldProps = {
  label: string;
  description?: string;
  children: React.ReactNode; // your <Select> goes here
};

export function SelectField({ label, description, children }: SelectFieldProps) {
  return (
    <div className="space-y-2">
      {/* Label + optional tooltip (same pattern as FileUpload) */}
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

      {children}
    </div>
  );
}
