"""
TrefzRoute Backend - FastAPI + Google OR-Tools VRP
Zeitfenster: erste Zustellung ab 07:30, letzte bis 16:30 (strikt)
Abfahrt: so spaet wie moeglich (Fahrzeit zum ersten Stopp - 07:30)
Rueckfahrt: keine Zeitbeschraenkung
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import math
import os

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_OK = True
    print("[OK] OR-Tools geladen")
except ImportError as e:
    ORTOOLS_OK = False
    print("[WARNUNG] OR-Tools nicht verfuegbar:", e)

app = FastAPI(title="TrefzRoute API", version="3.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -- ZEITKONSTANTEN --
EARLIEST_DELIVERY  = 7 * 3600 + 30 * 60   # 07:30 in Sekunden ab Mitternacht
LATEST_DELIVERY    = 16 * 3600 + 30 * 60   # 16:30 in Sekunden ab Mitternacht
EARLIEST_DEPARTURE = 6 * 3600              # 06:00 früheste Abfahrt
# Depot-Zeitfenster: [0, LATEST_DELIVERY + 3h] (Rückkehr bis 19:30 erlaubt)
LATEST_RETURN      = LATEST_DELIVERY + 3 * 3600  # 19:30

# -- MODELS --

class MatrixOptimizeRequest(BaseModel):
    addresses: List[str]
    dist_matrix: List[List[float]]
    dur_matrix: List[List[float]]         # inkl. Abladezeit — für OR-Tools Optimierung
    travel_dur_matrix: Optional[List[List[float]]] = None  # nur Fahrzeit — für Anzeige
    max_stops_per_route: int = 999
    max_duration_s: Optional[float] = None

class StopResult(BaseModel):
    address: str
    leg_dist_m: float
    leg_dist_text: str
    leg_dur_s: float
    leg_dur_text: str
    arrival_time: Optional[str] = None  # geschätzte Ankunftszeit

class RouteResult(BaseModel):
    id: int
    stops: List[StopResult]
    total_dist_m: float
    total_dist_text: str
    total_dur_s: float
    total_dur_text: str
    departure_time: Optional[str] = None
    return_time: Optional[str] = None
    feasible: bool = True

class OptimizeResponse(BaseModel):
    routes: List[RouteResult]
    total_dist_m: float
    total_dur_s: float
    num_stops: int
    solver: str
    warnings: List[str] = []

# -- HELPERS --

def format_dist(m: float) -> str:
    return f"{round(m)} m" if m < 1000 else f"{m/1000:.1f} km"

def format_dur(s: float) -> str:
    h, m = int(s // 3600), round((s % 3600) / 60)
    return f"{h}h {m}min" if h > 0 else f"{m} min"

def format_time(s: float) -> str:
    """Sekunden ab Mitternacht -> HH:MM"""
    h = int(s // 3600) % 24
    m = int((s % 3600) // 60)
    return f"{h:02d}:{m:02d}"

# -- GREEDY FALLBACK mit Zeitfenstern --

def solve_greedy(dist_matrix, dur_matrix, depot_idx=0):
    n = len(dist_matrix)
    unvisited = list(range(1, n))
    routes = []

    while unvisited:
        route = []
        current = depot_idx
        # Berechne Abfahrtszeit: so dass erste Zustellung um 07:30
        # Wir starten mit dem nächsten Stopp vom Depot und berechnen rückwärts
        current_time = EARLIEST_DEPARTURE  # starte mit 06:00 als Basis

        while unvisited:
            # Finde nächsten erreichbaren Stopp (Ankunft <= 16:30)
            candidates = []
            for s in unvisited:
                travel = dur_matrix[current][s]
                arrival = max(current_time + travel, EARLIEST_DELIVERY)
                if arrival <= LATEST_DELIVERY:
                    candidates.append((s, arrival))

            if not candidates:
                break

            # Nearest neighbor unter den erreichbaren
            nearest, arrival = min(candidates, key=lambda x: dist_matrix[current][x[0]])
            route.append((nearest, arrival))
            current_time = arrival
            unvisited.remove(nearest)
            current = nearest

        if route:
            routes.append(route)

    return routes

# -- OR-TOOLS VRP MIT ZEITFENSTERN --

def solve_ortools_vrp(dist_matrix, dur_matrix, depot_idx=0):
    n = len(dist_matrix)
    num_stops = n - 1

    # Fahrzeuganzahl: großzügig schätzen, OR-Tools minimiert selbst
    # Worst case: jeder Stopp braucht eigene Route (bei sehr langen Fahrten)
    # Typisch: 1 Fahrzeug pro ~8-10 Stopps bei 9h Fenster + 15min Abladen
    num_vehicles = min(num_stops, max(4, math.ceil(num_stops / 8)))
    print(f"[INFO] {num_stops} Stopps, {num_vehicles} max. Fahrzeuge, Zeitfenster 07:30-16:30")

    idist = [[int(dist_matrix[i][j]) for j in range(n)] for i in range(n)]
    idur  = [[int(dur_matrix[i][j])  for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    # Distanz-Callback
    def dist_cb(fi, ti):
        return idist[manager.IndexToNode(fi)][manager.IndexToNode(ti)]
    dist_ci = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_ci)

    # Zeit-Callback (Fahrtzeit + Abladezeit bereits enthalten)
    def time_cb(fi, ti):
        return idur[manager.IndexToNode(fi)][manager.IndexToNode(ti)]
    time_ci = routing.RegisterTransitCallback(time_cb)

    # Zeit-Dimension mit Zeitfenstern
    # Horizon: von 06:00 (Abfahrt) bis 19:30 (späteste Rückkehr)
    horizon = LATEST_RETURN
    routing.AddDimension(time_ci, 3600, horizon, False, "Time")
    # False = kein "force start cumul to zero" — erlaubt flexible Startzeiten
    time_dim = routing.GetDimensionOrDie("Time")

    # Zeitfenster für jeden Stopp: 07:30 – 16:30
    for i in range(1, n):  # alle außer Depot
        idx = manager.NodeToIndex(i)
        time_dim.CumulVar(idx).SetRange(EARLIEST_DELIVERY, LATEST_DELIVERY)

    # Depot: Abfahrt frühestens 06:00, Rückkehr spätestens 19:30
    for v in range(num_vehicles):
        start_idx = routing.Start(v)
        end_idx   = routing.End(v)
        time_dim.CumulVar(start_idx).SetRange(EARLIEST_DEPARTURE, LATEST_DELIVERY)
        time_dim.CumulVar(end_idx).SetRange(0, LATEST_RETURN)

    # Minimiere Gesamtdistanz (primary) und Fahrzeuganzahl (secondary)
    routing.SetFixedCostOfAllVehicles(100_000)

    # Suchparameter — 8s reichen für ~35 Stopps pro Cluster
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 8
    params.log_search = False

    sol = routing.SolveWithParameters(params)

    if not sol:
        # Schneller Retry mit anderer Strategie
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        params.time_limit.seconds = 5
        sol = routing.SolveWithParameters(params)

    if not sol:
        return None, manager, routing, None

    return sol, manager, routing, time_dim

# -- ENDPOINTS --

@app.get("/health")
def health():
    return {
        "status": "ok",
        "solver": "OR-Tools VRP + Zeitfenster" if ORTOOLS_OK else "Greedy",
        "ortools_available": ORTOOLS_OK,
        "version": "3.1.0"
    }

@app.post("/optimize_matrix", response_model=OptimizeResponse)
def optimize_matrix(req: MatrixOptimizeRequest):
    n = len(req.addresses)
    if n < 2:
        raise HTTPException(400, "Mindestens 1 Stopp erforderlich.")

    dist_matrix  = req.dist_matrix
    dur_matrix   = req.dur_matrix
    # Für Anzeige: reine Fahrzeit ohne Abladezeit
    travel_matrix = req.travel_dur_matrix if req.travel_dur_matrix else req.dur_matrix
    results, total_dist, total_dur, warnings = [], 0, 0, []

    if ORTOOLS_OK:
        try:
            sol, manager, routing, time_dim = solve_ortools_vrp(dist_matrix, dur_matrix)

            if sol is None:
                # Fallback auf Greedy
                raise Exception("Keine Loesung gefunden")

            solver_name = "OR-Tools VRP + Zeitfenster 07:30-16:30"
            num_vehicles = routing.vehicles()

            for v in range(num_vehicles):
                route_indices = []
                arrivals = []
                idx = routing.Start(v)
                while not routing.IsEnd(idx):
                    node = manager.IndexToNode(idx)
                    if node != 0:
                        arrival_s = sol.Value(time_dim.CumulVar(idx))
                        route_indices.append(node)
                        arrivals.append(arrival_s)
                    idx = sol.Value(routing.NextVar(idx))

                if not route_indices:
                    continue

                # Abfahrtszeit: rückwärts vom ersten Stopp berechnen
                # Nur reine Fahrzeit — Abladezeit gehört nicht zur Abfahrtsberechnung
                travel_to_first = int(travel_matrix[0][route_indices[0]])
                departure_s = max(EARLIEST_DEPARTURE, arrivals[0] - travel_to_first)

                stops_out, route_dist, route_dur, prev = [], 0, 0, 0
                for i, node in enumerate(route_indices):
                    leg_dist_m   = dist_matrix[prev][node]
                    leg_travel_s = travel_matrix[prev][node]  # reine Fahrzeit für Anzeige
                    leg_dur_s    = dur_matrix[prev][node]     # inkl. Abladezeit für Gesamtdauer
                    route_dist += leg_dist_m
                    route_dur  += leg_dur_s
                    stops_out.append(StopResult(
                        address=req.addresses[node],
                        leg_dist_m=leg_dist_m,
                        leg_dist_text=format_dist(leg_dist_m),
                        leg_dur_s=leg_travel_s,
                        leg_dur_text=format_dur(leg_travel_s),
                        arrival_time=format_time(arrivals[i]),
                    ))
                    prev = node

                # Rückfahrt: letzte Ankunftszeit + Fahrzeit zurück zum Depot
                last_arrival_s = arrivals[-1] if arrivals else departure_s
                return_dist_m = dist_matrix[prev][0]
                return_dur_s  = dur_matrix[prev][0]
                # Rückkehr-Uhrzeit = letzte Ankunft + Abladezeit bereits in dur enthalten
                # dur_matrix[prev][0] enthält keine Abladezeit (j=0 ist Depot)
                return_time_s = last_arrival_s + return_dur_s

                route_dist += return_dist_m
                route_dur  += return_dur_s
                total_dist += route_dist
                total_dur  += route_dur

                # Prüfe ob letzte Zustellung vor 16:30
                last_arrival = arrivals[-1] if arrivals else 0
                feasible = last_arrival <= LATEST_DELIVERY
                if not feasible:
                    warnings.append(f"Route {len(results)+1}: letzte Zustellung {format_time(last_arrival)} > 16:30")

                results.append(RouteResult(
                    id=len(results) + 1,
                    stops=stops_out,
                    total_dist_m=route_dist,
                    total_dist_text=format_dist(route_dist),
                    total_dur_s=route_dur,
                    total_dur_text=format_dur(route_dur),
                    departure_time=format_time(departure_s),
                    return_time=format_time(return_time_s),
                    feasible=feasible,
                ))

        except Exception as e:
            print(f"OR-Tools Fehler: {e} — Greedy Fallback")
            greedy_routes = solve_greedy(dist_matrix, dur_matrix)
            solver_name = "Greedy + Zeitfenster (Fallback)"

            for ri, route in enumerate(greedy_routes):
                stops_out, route_dist, route_dur, prev = [], 0, 0, 0
                departure_s = None
                last_arrival_s = 0
                for node, arrival_s in route:
                    leg_dist_m   = dist_matrix[prev][node]
                    leg_travel_s = travel_matrix[prev][node]
                    leg_dur_s    = dur_matrix[prev][node]
                    if departure_s is None:
                        departure_s = max(EARLIEST_DEPARTURE, arrival_s - leg_travel_s)
                    route_dist += leg_dist_m
                    route_dur  += leg_dur_s
                    last_arrival_s = arrival_s
                    stops_out.append(StopResult(
                        address=req.addresses[node],
                        leg_dist_m=leg_dist_m,
                        leg_dist_text=format_dist(leg_dist_m),
                        leg_dur_s=leg_travel_s,
                        leg_dur_text=format_dur(leg_travel_s),
                        arrival_time=format_time(arrival_s),
                    ))
                    prev = node
                return_dur_s = dur_matrix[prev][0]
                return_time_s = last_arrival_s + return_dur_s
                route_dist += dist_matrix[prev][0]
                route_dur  += return_dur_s
                total_dist += route_dist
                total_dur  += route_dur
                results.append(RouteResult(
                    id=ri + 1, stops=stops_out,
                    total_dist_m=route_dist, total_dist_text=format_dist(route_dist),
                    total_dur_s=route_dur,  total_dur_text=format_dur(route_dur),
                    departure_time=format_time(departure_s) if departure_s else None,
                    return_time=format_time(return_time_s),
                    feasible=True,
                ))
    else:
        greedy_routes = solve_greedy(dist_matrix, dur_matrix)
        solver_name = "Greedy + Zeitfenster"
        for ri, route in enumerate(greedy_routes):
            stops_out, route_dist, route_dur, prev = [], 0, 0, 0
            for node, arrival_s in route:
                leg_dist_m   = dist_matrix[prev][node]
                leg_travel_s = travel_matrix[prev][node]
                leg_dur_s    = dur_matrix[prev][node]
                route_dist += leg_dist_m
                route_dur  += leg_dur_s
                stops_out.append(StopResult(
                    address=req.addresses[node],
                    leg_dist_m=leg_dist_m, leg_dist_text=format_dist(leg_dist_m),
                    leg_dur_s=leg_travel_s, leg_dur_text=format_dur(leg_travel_s),
                    arrival_time=format_time(arrival_s),
                ))
                prev = node
            route_dist += dist_matrix[prev][0]
            route_dur  += dur_matrix[prev][0]
            total_dist += route_dist
            total_dur  += route_dur
            results.append(RouteResult(
                id=ri + 1, stops=stops_out,
                total_dist_m=route_dist, total_dist_text=format_dist(route_dist),
                total_dur_s=route_dur,  total_dur_text=format_dur(route_dur),
                feasible=True,
            ))

    return OptimizeResponse(
        routes=results, total_dist_m=total_dist, total_dur_s=total_dur,
        num_stops=n - 1, solver=solver_name, warnings=warnings,
    )

# -- STATIC FRONTEND --
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
