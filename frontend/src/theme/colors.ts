/**
 * Centralized color palette for the entire application.
 *
 * NEUTRAL — UI backgrounds, text, borders, structural elements.
 * TRAFFIC — contrast colors for density visualization, highlights, overlays.
 */

export const NEUTRAL = {
  black:       "#000000",
  forest:      "#3A4442",
  warmSlate:   "#423E3A",
  midGray:     "#545454",
  steel:       "#5E6762",
  slate:       "#70747D",
  sageGray:    "#878D85",
  silver:      "#A6A6A6",
  warmSilver:  "#B2B4AB",
  linen:       "#DFDCD3",
  cream:       "#E8E4D9",
  white:       "#FFFFFF",
} as const;

export const TRAFFIC = {
  green:       { light: "#2ECC71", dark: "#27AE60" },
  yellow:      { light: "#F1C40F", dark: "#D4AC0D" },
  orange:      { light: "#E67E22", dark: "#CA6F1E" },
  red:         { light: "#E74C3C", dark: "#C0392B" },
  violet:      { light: "#8E44AD", dark: "#6C3483" },
  teal:        { light: "#1ABC9C", dark: "#148F77" },
} as const;
