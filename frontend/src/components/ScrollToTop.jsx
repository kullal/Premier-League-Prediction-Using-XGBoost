import { useEffect } from "react";
import { useLocation } from "react-router-dom";

export default function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    if (typeof window === "undefined" || !window.scrollTo) return;

    const prefersReduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (prefersReduced) {
      requestAnimationFrame(() => window.scrollTo({ top: 0, left: 0, behavior: "auto" }));
      return;
    }

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setTimeout(() => {
          try {
            window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
          } catch (e) {
            window.scrollTo(0, 0);
          }
        }, 50);
      });
    });
  }, [pathname]);

  return null;
}
