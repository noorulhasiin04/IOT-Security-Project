import { useState, useEffect } from 'react';

/**
 * Custom Hook: useLocalStorage
 * Synchronizes React state with browser localStorage.
 * 
 * @param {string} key - The key name in localStorage
 * @param {any} initialValue - Default fallback value if key does not exist
 * @returns {[any, Function]} - Returns current stored state and a setter function
 */
export function useLocalStorage(key, initialValue) {
  // Initialize state from localStorage or fallback to initialValue
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  // Keep localStorage updated whenever storedValue changes
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(storedValue));
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setStoredValue];
}
