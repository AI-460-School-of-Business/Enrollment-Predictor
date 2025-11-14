import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { Checkbox } from './ui/checkbox';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Button } from './ui/button';

interface SemesterSelectorProps {
  selectedSemesters: string[];
  onSelectionChange: (semesters: string[]) => void;
}

export function SemesterSelector({ selectedSemesters, onSelectionChange }: SemesterSelectorProps) {
  const years = [2022, 2023, 2024, 2025];
  const seasons = ['Fall', 'Spring'];
  
  const allSemesters = years.flatMap(year => 
    seasons.map(season => `${season} ${year}`)
  );

  const handleToggle = (semester: string) => {
    if (selectedSemesters.includes(semester)) {
      onSelectionChange(selectedSemesters.filter(s => s !== semester));
    } else {
      onSelectionChange([...selectedSemesters, semester]);
    }
  };

  return (
    <div className="space-y-2">
      <label className="text-sm">Semesters</label>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className="w-full justify-between border-gray-300 hover:border-[#94BAEB]"
          >
            <span className="text-sm">
              {selectedSemesters.length > 0
                ? `${selectedSemesters.length} selected`
                : 'Select semesters'}
            </span>
            <ChevronDown className="w-4 h-4 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-64 p-4" align="start">
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {allSemesters.map((semester) => (
              <div key={semester} className="flex items-center space-x-2">
                <Checkbox
                  id={semester}
                  checked={selectedSemesters.includes(semester)}
                  onCheckedChange={() => handleToggle(semester)}
                />
                <label
                  htmlFor={semester}
                  className="text-sm cursor-pointer flex-1"
                >
                  {semester}
                </label>
              </div>
            ))}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
