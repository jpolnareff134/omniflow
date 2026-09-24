package org.dacapo.omniflow.agent;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

final class CycleMarkers {
    private CycleMarkers() { }

    static List<Long> load(String path) {
        if (path == null || path.trim().isEmpty()) return Collections.emptyList();
        File file = new File(path);
        if (!file.isFile()) throw new IllegalArgumentException("cycle markers file not found: " + path);
        List<Long> markers = new ArrayList<Long>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            long previous = -1L;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                long marker = Long.parseLong(line);
                if (marker < 0L || marker <= previous) {
                    reader.close();
                    throw new IllegalArgumentException(
                            "cycle markers must be non-negative and strictly increasing: " + path);
                }
                markers.add(Long.valueOf(marker));
                previous = marker;
            }
            reader.close();
        } catch (IOException error) {
            throw new IllegalArgumentException("cannot read cycle markers file: " + path, error);
        }
        return Collections.unmodifiableList(markers);
    }
}
